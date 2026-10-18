package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Config is the part of config.json the forward pass needs.
type Config struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	Intermediate int     `json:"intermediate_size"`
	MaxPositions int     `json:"max_position_embeddings"`
	Vocab        int     `json:"vocab_size"`
	Eps          float64 `json:"layer_norm_eps"`
	Act          string  `json:"hidden_act"`
}

type linear struct {
	w   []float32 // out × in, row-major (PyTorch's layout)
	b   []float32
	in  int
	out int
}

type layerNorm struct{ w, b []float32 }

type layer struct {
	q, k, v, o    linear
	attnNorm      layerNorm
	inter, output linear
	outNorm       layerNorm
}

// Model is a BERT encoder in float32 (as candle runs it on the CPU).
type Model struct {
	cfg            Config
	word, pos, typ []float32
	embNorm        layerNorm
	layers         []layer
}

// tensors maps a safetensors file (an 8-byte header length, a JSON header
// of name → {dtype, shape, data_offsets}, then the data) read-only. F32
// tensors are views of the mapping where they are aligned on a
// little-endian machine (every release target), so nothing is copied and
// every atlas-cli process shares the file's pages; otherwise, or where the
// file cannot be mapped, they are copied out. A view outlives the close and
// the file's removal: Download replaces files by rename, never in place.
// release unmaps the file, for a caller that will not use the tensors.
func tensors(path string) (_ map[string][]float32, _ func(), err error) {
	b, mapped, err := mapFile(path)
	if err != nil {
		return nil, nil, err
	}
	views := false
	unmap := func() {
		if mapped {
			_ = unix.Munmap(b)
			mapped = false
		}
	}
	defer func() {
		if err != nil || !views {
			unmap()
		}
	}()
	if len(b) < 8 {
		return nil, nil, fmt.Errorf("%s: too short", path)
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n > uint64(len(b)-8) { //nolint:gosec // G115: len(b) ≥ 8 here
		return nil, nil, fmt.Errorf("%s: bad header", path)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &hdr); err != nil {
		return nil, nil, err
	}
	data := b[8+n:]
	out := map[string][]float32{}
	for _, name := range slices.Sorted(maps.Keys(hdr)) { // the same first error each time
		raw := hdr[name]
		if name == "__metadata__" {
			continue
		}
		var t struct {
			Dtype   string   `json:"dtype"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &t); err != nil {
			return nil, nil, err
		}
		if t.Dtype != "F32" {
			continue // position_ids (I64) and the like
		}
		lo, hi := t.Offsets[0], t.Offsets[1]
		if lo < 0 || hi > int64(len(data)) || lo > hi || (hi-lo)%4 != 0 {
			return nil, nil, fmt.Errorf("%s: bad offsets for %s", path, name)
		}
		if lo == hi {
			out[name] = []float32{}
			continue
		}
		p := unsafe.Pointer(&data[lo]) //nolint:gosec // G103: a view of the read-only mapping, checked for alignment
		if mapped && littleEndian && uintptr(p)%4 == 0 {
			out[name] = unsafe.Slice((*float32)(p), (hi-lo)/4) //nolint:gosec // G103: as above
			views = true
			continue
		}
		f := make([]float32, (hi-lo)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[lo+int64(i)*4:]))
		}
		out[name] = f
	}
	return out, unmap, nil
}

// mapFile maps path read-only, or reads it where it cannot be mapped.
func mapFile(path string) ([]byte, bool, error) {
	f, err := os.Open(path) //nolint:gosec // G304: the model's own file
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if size := fi.Size(); size > 0 && size == int64(int(size)) {
		if b, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED); err == nil {
			return b, true, nil
		}
	}
	b, err := io.ReadAll(f)
	return b, false, err
}

var littleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1 //nolint:gosec // G103: reads the first byte of a local
}()

// LoadModel reads config.json and model.safetensors.
func LoadModel(configPath, weightsPath string) (*Model, error) {
	cb, err := os.ReadFile(configPath) //nolint:gosec // G304: the model's own file
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return nil, err
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-12
	}
	if cfg.Hidden == 0 || cfg.Heads == 0 || cfg.Hidden%cfg.Heads != 0 {
		return nil, fmt.Errorf("config.json: bad sizes")
	}
	ts, release, err := tensors(weightsPath)
	if err != nil {
		return nil, err
	}
	var missing string
	get := func(name string, size int) []float32 {
		for _, n := range []string{name, "bert." + name} {
			if t, ok := ts[n]; ok && len(t) == size {
				return t
			}
		}
		if missing == "" {
			missing = name
		}
		return make([]float32, size)
	}
	h, in := cfg.Hidden, cfg.Intermediate
	lin := func(p string, inDim, outDim int) linear {
		return linear{w: get(p+".weight", inDim*outDim), b: get(p+".bias", outDim), in: inDim, out: outDim}
	}
	ln := func(p string) layerNorm { return layerNorm{w: get(p+".weight", h), b: get(p+".bias", h)} }
	m := &Model{cfg: cfg,
		word:    get("embeddings.word_embeddings.weight", cfg.Vocab*h),
		pos:     get("embeddings.position_embeddings.weight", cfg.MaxPositions*h),
		typ:     get("embeddings.token_type_embeddings.weight", 2*h),
		embNorm: ln("embeddings.LayerNorm"),
	}
	for i := range cfg.Layers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.layers = append(m.layers, layer{
			q: lin(p+"attention.self.query", h, h), k: lin(p+"attention.self.key", h, h), v: lin(p+"attention.self.value", h, h),
			o: lin(p+"attention.output.dense", h, h), attnNorm: ln(p + "attention.output.LayerNorm"),
			inter: lin(p+"intermediate.dense", h, in), output: lin(p+"output.dense", in, h), outNorm: ln(p + "output.LayerNorm"),
		})
	}
	if missing != "" {
		release() // Default loads again on its next call: no mapping per try
		return nil, fmt.Errorf("model.safetensors: no tensor %s", missing)
	}
	return m, nil
}

// Dim is the embedding size.
func (m *Model) Dim() int { return m.cfg.Hidden }

// apply computes y = x·Wᵀ + b for n rows, the outputs split across cores.
// Rows go in blocks that stay in cache, four at a time against each weight
// row, so a weight is loaded once per four products.
func (l linear) apply(x []float32, n int) []float32 {
	y := make([]float32, n*l.out)
	in := l.in
	parallel(l.out, 32, func(lo, hi int) {
		for r0 := 0; r0 < n; r0 += rowBlock {
			r1 := min(r0+rowBlock, n)
			for o := lo; o < hi; o++ {
				wr := l.w[o*in : (o+1)*in]
				bias := l.b[o]
				r := r0
				for ; r+4 <= r1; r += 4 {
					a, b, c, d := dot4(wr, x[r*in:(r+1)*in], x[(r+1)*in:(r+2)*in], x[(r+2)*in:(r+3)*in], x[(r+3)*in:(r+4)*in])
					y[r*l.out+o] = a + bias
					y[(r+1)*l.out+o] = b + bias
					y[(r+2)*l.out+o] = c + bias
					y[(r+3)*l.out+o] = d + bias
				}
				for ; r < r1; r++ {
					y[r*l.out+o] = dot(x[r*in:(r+1)*in], wr) + bias
				}
			}
		}
	})
	return y
}

// rowBlock rows of a 384-wide input are 96 KB: they stay in L2 while every
// output of a chunk passes over them.
const rowBlock = 64

// dot4 is w·a, w·b, w·c and w·d in one pass over w.
func dot4(w, a, b, c, d []float32) (float32, float32, float32, float32) {
	a, b, c, d = a[:len(w)], b[:len(w)], c[:len(w)], d[:len(w)]
	var sa, sb, sc, sd float32
	for i, x := range w {
		sa += x * a[i]
		sb += x * b[i]
		sc += x * c[i]
		sd += x * d[i]
	}
	return sa, sb, sc, sd
}

// threads caps the cores one embedding uses; 0 is every core the process
// may use.
var threads atomic.Int32

// SetThreads sets the cap (0 lifts it).
func SetThreads(n int) { threads.Store(int32(max(n, 0))) } //nolint:gosec // G115: a core count

// workers is how many goroutines one step of a forward pass runs on.
func workers() int {
	n := runtime.GOMAXPROCS(0)
	if t := int(threads.Load()); t > 0 && t < n {
		return t
	}
	return n
}

// parallel runs fn over [0, n) in at most workers() contiguous chunks of
// at least minChunk, inline when that is one.
func parallel(n, minChunk int, fn func(lo, hi int)) {
	w := min(workers(), max(n/minChunk, 1))
	if w <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, min(lo+chunk, n))
		}()
	}
	wg.Wait()
}

func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func (ln layerNorm) apply(x []float32, n, h int, eps float64) {
	for r := range n {
		row := x[r*h : (r+1)*h]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(h)
		var vr float64
		for _, v := range row {
			d := float64(v) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(h)+eps)
		for i, v := range row {
			row[i] = float32((float64(v)-mean)*inv)*ln.w[i] + ln.b[i]
		}
	}
}

func gelu(x float32) float32 {
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
}

// Forward runs the encoder on one sequence and returns the mean of its
// token states (every token is real: one text, no padding), L2-normalised.
func (m *Model) Forward(ids []int32) []float32 {
	h, n := m.cfg.Hidden, len(ids)
	if n == 0 {
		return make([]float32, h)
	}
	x := make([]float32, n*h)
	for t, id := range ids {
		if int(id) >= m.cfg.Vocab || t >= m.cfg.MaxPositions {
			continue
		}
		row := x[t*h : (t+1)*h]
		for i := range h {
			row[i] = m.word[int(id)*h+i] + m.pos[t*h+i] + m.typ[i]
		}
	}
	eps := m.cfg.Eps
	m.embNorm.apply(x, n, h, eps)
	heads := m.cfg.Heads
	hd := h / heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	for _, L := range m.layers {
		q, k, v := L.q.apply(x, n), L.k.apply(x, n), L.v.apply(x, n)
		ctx := make([]float32, n*h)
		parallel(heads, 1, func(lo, hi int) {
			scores := make([]float32, n)
			for hh := lo; hh < hi; hh++ {
				for i := range n {
					qi := q[i*h+hh*hd : i*h+hh*hd+hd]
					maxS := float32(math.Inf(-1))
					for j := range n {
						s := dot(qi, k[j*h+hh*hd:j*h+hh*hd+hd]) * scale
						scores[j] = s
						maxS = max(maxS, s)
					}
					var sum float32
					for j := range n {
						scores[j] = float32(math.Exp(float64(scores[j] - maxS)))
						sum += scores[j]
					}
					out := ctx[i*h+hh*hd : i*h+hh*hd+hd]
					for j := range n {
						p := scores[j] / sum
						vj := v[j*h+hh*hd : j*h+hh*hd+hd]
						for d := range hd {
							out[d] += p * vj[d]
						}
					}
				}
			}
		})
		a := L.o.apply(ctx, n)
		for i := range a {
			a[i] += x[i]
		}
		L.attnNorm.apply(a, n, h, eps)
		inter := L.inter.apply(a, n)
		for i := range inter {
			inter[i] = gelu(inter[i])
		}
		out := L.output.apply(inter, n)
		for i := range out {
			out[i] += a[i]
		}
		L.outNorm.apply(out, n, h, eps)
		x = out
	}
	pooled := make([]float32, h)
	for t := range n {
		for i := range h {
			pooled[i] += x[t*h+i]
		}
	}
	var norm float64
	for i := range pooled {
		pooled[i] /= float32(n)
		norm += float64(pooled[i]) * float64(pooled[i])
	}
	norm = math.Max(math.Sqrt(norm), 1e-12)
	for i := range pooled {
		pooled[i] = float32(float64(pooled[i]) / norm)
	}
	return pooled
}
