package embed

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

// writeSafetensors writes tensors as a safetensors file whose header is
// padded so the data starts at offset 8+hdr+pad.
func writeSafetensors(t *testing.T, ts map[string][]float32, pad int) string {
	t.Helper()
	hdr := map[string]any{"__metadata__": map[string]string{"format": "pt"},
		"position_ids": map[string]any{"dtype": "I64", "shape": []int{1}, "data_offsets": []int{0, 8}}}
	var data []byte
	data = binary.LittleEndian.AppendUint64(data, 7)
	for _, name := range []string{"a", "b"} {
		lo := len(data)
		for _, x := range ts[name] {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(x))
		}
		hdr[name] = map[string]any{"dtype": "F32", "shape": []int{len(ts[name])}, "data_offsets": []int{lo, len(data)}}
	}
	h, _ := json.Marshal(hdr)
	for range pad {
		h = append(h, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(append(out, h...), data...)
	p := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(p, out, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTensorsMappedAndCopied(t *testing.T) {
	want := map[string][]float32{"a": {1, -2.5, 3e-7}, "b": {0.25, 1e9}}
	// Each padding puts the data at another alignment: views where it is
	// aligned, copies where it is not, the same values either way.
	for pad := range 4 {
		got, _, err := tensors(writeSafetensors(t, want, pad))
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := got["position_ids"]; ok {
			t.Fatal("an I64 tensor was read as F32")
		}
		for name, w := range want {
			if len(got[name]) != len(w) {
				t.Fatalf("pad %d: %s has %d values", pad, name, len(got[name]))
			}
			for i := range w {
				if got[name][i] != w[i] {
					t.Fatalf("pad %d: %s[%d] = %v, want %v", pad, name, i, got[name][i], w[i])
				}
			}
		}
	}
}

func TestTensorsRejectsABadFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.safetensors")
	_ = os.WriteFile(p, []byte{0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, '{'}, 0o644)
	if _, _, err := tensors(p); err == nil {
		t.Fatal("a header past the end was accepted")
	}
}

// The blocked kernel matches a plain y = x·Wᵀ + b for row counts around
// the four-row step and the row block, on one core and on many.
func TestLinearApply(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	in, out := 37, 70
	l := linear{w: make([]float32, in*out), b: make([]float32, out), in: in, out: out}
	for i := range l.w {
		l.w[i] = rng.Float32()*2 - 1
	}
	for i := range l.b {
		l.b[i] = rng.Float32()
	}
	for _, threads := range []int{1, 0} {
		SetThreads(threads)
		for _, n := range []int{1, 3, 4, 5, rowBlock - 1, rowBlock, rowBlock + 5, 2*rowBlock + 3} {
			x := make([]float32, n*in)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			y := l.apply(x, n)
			for r := range n {
				for o := range out {
					var want float64
					for i := range in {
						want += float64(x[r*in+i]) * float64(l.w[o*in+i])
					}
					want += float64(l.b[o])
					if d := math.Abs(float64(y[r*out+o]) - want); d > 1e-4 {
						t.Fatalf("threads %d, n %d: y[%d][%d] = %v, want %v", threads, n, r, o, y[r*out+o], want)
					}
				}
			}
		}
	}
	SetThreads(0)
}

func TestParallelCoversEveryIndexOnce(t *testing.T) {
	for _, threads := range []int{1, 3, 0} {
		SetThreads(threads)
		for _, n := range []int{0, 1, 7, 100, 1536} {
			seen := make([]int32, n)
			parallel(n, 1, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					seen[i]++
				}
			})
			for i, c := range seen {
				if c != 1 {
					t.Fatalf("threads %d, n %d: index %d seen %d times", threads, n, i, c)
				}
			}
		}
	}
	SetThreads(0)
}
