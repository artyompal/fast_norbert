# Fast Norbert

Fast Norbert is an optimized fork of https://github.com/sigsep/norbert.

## Performance

This is time (in seconds) that the filtering process takes on a single core:

| Test case | Original Norbert | Fast Norbert |
| ----------| ---------------- |--------------| 
| song 1 | 19.3 | **7.5** |
| song 2 | 27.5 | **10.9** |

# Norbert filter

Wiener filter is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation.

This filtering method assumes you have some way of estimating power or magnitude spectrograms for all the audio sources (non-negative) composing a mixture. If you only have a model for some _target_ sources, and not for the rest, you may use `fast_norbert.residual_model` to let Norbert create a residual model for you.

Given all source spectrograms and the mixture Time-Frequency representation, this repository can build and apply the filter that is appropriate for separation, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called _Expectation Maximization_, where filtering and re-estimation of the parameters are iterated.

From a beginner's perspective, all you need to do is often to call `fast_norbert.wiener` with the mix and your spectrogram estimates. This should handle the rest.

From a more expert perspective, you will find the different ingredients from the EM algorithm as functions in the module `fast_norbert.norbert`.

## Installation

`pip install fast_norbert`

## Usage

Asssuming a complex spectrogram `X`, and a (magnitude) estimate of a target to be extracted from the spectrogram, performing the multichannel wiener filter is as simple as this:

```python
import fast_norbert

x = stft(audio)
v = model(x)
y = fast_norbert.wiener(v, x)
estimate = istft(y)
```

## Authors

Artyom Palvelev (this repo) <br>
Antoine Liutkus, Fabian-Robert Stöter (original repo)

## License

MIT
