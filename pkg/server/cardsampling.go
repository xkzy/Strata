package server

import (
	"strata/pkg/chattemplate"
	"strata/pkg/engineipc"
)

// thinkingOn reports whether the template will open a thinking block for these options (its default is on).
func thinkingOn(o chattemplate.Options) bool { return o.EnableThinking == nil || *o.EnableThinking }

// cardSampling is the sampling the Qwen3.8-Flash-Next model card recommends, by thinking state. It is used when neither
// the request nor the config's "sampling" block says otherwise; greedy decoding is what makes small local models loop.
// min_p 0 and repetition_penalty 1 are the engine's own "off", so they are left unset.
func cardSampling(thinking bool) engineipc.SamplingParams {
	if thinking {
		return engineipc.SamplingParams{Temperature: 1.0, TopP: 0.95, TopK: 20}
	}
	return engineipc.SamplingParams{Temperature: 0.7, TopP: 0.8, TopK: 20, PresencePenalty: 1.5}
}
