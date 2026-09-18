# Third-party code

This repository reruns and extends experiments from:

**Eliciting Secret Knowledge from Language Models**
Bartosz Cywiński, Emil Ryd, Rowan Wang, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy, Samuel Marks
arXiv:2510.01070 · https://github.com/cywinski/eliciting-secret-knowledge · MIT licence

The pipeline invoked by `scripts/` is theirs, checked out at commit `45aae57930a7d6ef5503a9b9b23e0b23fd8068ca`.
Scripts prefixed `mod_`, `k10_` and `greedy_` are generated at run time by `sed` from their originals;
the edits are only batch sizes, response counts and sampling temperature, all listed in the README's
deviations table. No upstream source file is redistributed here — `00_bootstrap.sh` clones it.

The model organisms (`bcywinski/gemma-2-9b-it-taboo-*`) and the base model (`google/gemma-2-9b-it`)
are downloaded from Hugging Face under their own licences; the Gemma models require accepting
Google's terms.

Files under `data/raw/` are outputs produced by running that pipeline: model responses from the
taboo organisms and guesses from the `gemma-3-4b-it` auditor.

## Upstream licence

MIT License · Copyright (c) 2025 Bartosz Cywiński

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
