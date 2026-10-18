package main

import (
	"fmt"
	"net"
	"net/url"
	"os"

	"github.com/spf13/cobra"

	"github.com/tryatlas/cli/internal/embed"
)

// atlas-cli memory compute: how memory search embeds on this machine.
func newMemoryComputeCmd() *cobra.Command {
	c := withRun(leaf("compute [auto|off|cpu|ollama|<url>]",
		"Show or set how memory search embeds here: automatic, keywords only, this CPU, or a GPU endpoint", cobra.MaximumNArgs(1)), runMemoryCompute)
	c.Long = `Show or set how memory search embeds on this machine (every Project here).

  auto     this machine's CPU, or keywords only on a small machine (one core,
           or under 2 GB); a local Ollama serving all-minilm is used when found
  off      keywords only: the model is never loaded
  cpu      this machine's CPU, whatever its size
  ollama   Ollama on this machine (ollama pull all-minilm)
  <url>    an OpenAI-compatible embeddings API, such as Hugging Face TEI or vLLM
           on a GPU (http://gpu-box:8080/v1)

An endpoint is used only once its vectors match the app's model. ATLAS_EMBED,
ATLAS_EMBED_MODEL and ATLAS_EMBED_THREADS override the setting.`
	c.Flags().String("model", "", "the model an endpoint is asked for")
	c.Flags().Int("threads", -1, "cores one embedding uses on the CPU (0: automatic)")
	return c
}

func runMemoryCompute(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	out := cmd.OutOrStdout()
	threads, _ := cmd.Flags().GetInt("threads")
	model, _ := cmd.Flags().GetString("model")
	if len(args) == 0 && threads < 0 && model == "" {
		printCompute(cmd, embed.Current())
		return nil
	}
	s := embed.SavedSettings()
	if len(args) == 1 {
		mode, err := embed.ParseMode(args[0])
		if err != nil {
			return err
		}
		s.Mode, s.Model = mode, ""
	}
	if model != "" {
		s.Model = model
	}
	if threads >= 0 {
		s.Threads = threads
	}
	p := embed.Choose(s, embed.DetectMachine())
	if p.Backend != embed.Endpoint {
		s.Model = "" // a model is an endpoint's
	}
	if p.Backend == embed.Endpoint {
		fmt.Fprintf(out, "Checking %s against the app's model…\n", endpointLabel(p))
		if err := embed.VerifyEndpoint(ctx, p.URL, p.Model); err != nil {
			if p.Mode == "ollama" {
				return fmt.Errorf("%s: %w; not saved (is Ollama running, with ollama pull %s done?)", endpointLabel(p), err, embed.OllamaModel)
			}
			return fmt.Errorf("%s: %w; not saved", endpointLabel(p), err)
		}
		if !loopback(p.URL) {
			fmt.Fprintf(out, "note: memory text is sent to %s to be embedded.\n", hostOf(p.URL))
		}
	}
	if err := embed.SaveSettings(s); err != nil {
		return err
	}
	if os.Getenv("ATLAS_EMBED") != "" {
		fmt.Fprintln(out, "note: ATLAS_EMBED is set in this environment and overrides the saved mode.")
	}
	p = embed.Current()
	printCompute(cmd, p)
	if p.Backend == embed.CPU && !embed.Available() {
		fmt.Fprintln(out, "atlas-cli memory index downloads the model ("+embed.Size+") and builds the index.")
	}
	return nil
}

func printCompute(cmd *cobra.Command, p embed.Plan) {
	w := cmd.OutOrStdout()
	fmt.Fprintf(w, "Machine:   %s\n", p.Machine)
	fmt.Fprintf(w, "Mode:      %s\n", p.Mode)
	fmt.Fprintf(w, "Search:    %s\n", describePlan(p))
}

// describePlan is a plan in one line: what search matches on, then where
// the vectors come from.
func describePlan(p embed.Plan) string {
	switch p.Backend {
	case embed.Endpoint:
		return "meaning and keywords · " + endpointLabel(p) + " · " + p.Note
	case embed.CPU:
		if err := embed.LoadError(); err != nil {
			return fmt.Sprintf("keywords only · the model files do not load (%v); atlas-cli memory index downloads them again", err)
		}
		if !embed.Available() {
			line := "keywords only · the model is not downloaded (atlas-cli memory index)"
			if p.Note != "" && p.Note != "as set" {
				line += " · " + p.Note
			}
			return line
		}
		cores := "every core"
		if p.Threads > 0 && p.Threads < p.Machine.CPUs {
			cores = fmt.Sprintf("%d cores", p.Threads)
		}
		line := fmt.Sprintf("meaning and keywords · this machine's CPU, %s (background passes: %d)", cores, p.Background)
		if p.Note != "" {
			line += " · " + p.Note
		}
		return line
	}
	return "keywords only · " + p.Note
}

func endpointLabel(p embed.Plan) string { return p.Model + " at " + hostOf(p.URL) }

func hostOf(u string) string {
	if pu, err := url.Parse(u); err == nil && pu.Host != "" {
		return pu.Host
	}
	return u
}

// loopback reports an endpoint on this machine.
func loopback(u string) bool {
	pu, err := url.Parse(u)
	if err != nil {
		return false
	}
	if pu.Hostname() == "localhost" {
		return true
	}
	ip := net.ParseIP(pu.Hostname())
	return ip != nil && ip.IsLoopback()
}
