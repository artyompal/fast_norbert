
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using complex = std::complex<double>;


#define USE_UNCHECKED 1

#if USE_UNCHECKED
# define cref(VAR, ...) VAR##_ref(__VA_ARGS__)
# define ref(VAR, ...) VAR##_ref(__VA_ARGS__)
#else
# define cref(VAR, ...) VAR.at(__VA_ARGS__)
# define ref(VAR, ...) VAR.mutable_at(__VA_ARGS__)
#endif


py::array_t<complex> get_mix_model(const py::array_t<double> &v_j, const py::array_t<complex> &r_j) {
/*
    Compute the model covariance of a mixture based on local Gaussian models.
    simply adds up all the v[..., j] * R[..., j]

    Parameters
    v_j: np.ndarray [shape=(nb_frames, nb_bins, nb_sources)] (nb_frames is usually 1)
        Power spectral densities for the sources

    r_j: np.ndarray [shape=(nb_bins, nb_channels, nb_channels, nb_sources)]
        Spatial covariance matrices of each sources

    Returns
    cxx: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        Covariance matrix for the mixture
*/
    size_t nb_channels = r_j.shape(1);
    size_t nb_frames = v_j.shape(0), nb_bins = v_j.shape(1), nb_sources = v_j.shape(2);

    py::array_t<complex> cxx({nb_frames, nb_bins, nb_channels, nb_channels});
    std::fill(cxx.mutable_data(), cxx.mutable_data() + cxx.size(), 0);

    auto v_j_ref = v_j.unchecked<3>();
    auto r_j_ref = r_j.unchecked<4>();
    auto cxx_ref = cxx.mutable_unchecked<4>();

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t source = 0; source < nb_sources; source++) {
                double a = cref(v_j, frame, bin, source);

                for (size_t i = 0; i < nb_channels; i++) {
                    for (size_t j = 0; j < nb_channels; j++) {
                        complex b = cref(r_j, bin, i, j, source);
                        ref(cxx, frame, bin, i, j) += a * b;
                    }
                }
            }
        }
    }

    return cxx;
}

py::array_t<complex> wiener_gain(const py::array_t<double> &v_j, const py::array_t<complex> &r_j,
                                 const py::array_t<complex> &inv_cxx) {
/*
    Compute the wiener gain for separating one source, given all parameters.
    It is the matrix applied to the mix to get the posterior mean of the source.

    Parameters
    v_j: np.ndarray [shape=(nb_frames, nb_bins)] (nb_frames is usually 1)
        power spectral density of the target source.

    r_j: np.ndarray [shape=(nb_bins, nb_channels, nb_channels)]
        spatial covariance matrix of the target source

    inv_cxx: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        inverse of the mixture covariance matrices

    Returns
    g: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        wiener filtering matrices, to apply to the mix, e.g. through
        `apply_filter` to get the target source estimate.
*/
    size_t nb_frames = v_j.shape(0), nb_bins = v_j.shape(1), nb_channels = r_j.shape(2);

    py::array_t<complex> g({nb_frames, nb_bins, nb_channels, nb_channels});
    std::fill(g.mutable_data(), g.mutable_data() + g.size(), 0);

    auto v_j_ref = v_j.unchecked<2>();
    auto r_j_ref = r_j.unchecked<3>();
    auto inv_cxx_ref = inv_cxx.unchecked<4>();
    auto g_ref = g.mutable_unchecked<4>();

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t i = 0; i < nb_channels; i++) {
                for (size_t j = 0; j < nb_channels; j++) {
                    complex delta = 0.0;

                    for (size_t k = 0; k < nb_channels; k++) {
                        delta += cref(r_j, bin, i, k) * cref(inv_cxx, frame, bin, k, j);
                    }

                    ref(g, frame, bin, i, j) += delta;
                }
            }
        }
    }

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            double v = cref(v_j, frame, bin);

            for (size_t i = 0; i < nb_channels; i++) {
                for (size_t j = 0; j < nb_channels; j++) {
                    ref(g, frame, bin, i, j) *= v;
                }
            }
        }
    }

    return g;
}

py::array_t<complex> apply_filter(const py::array_t<complex> &x, const py::array_t<complex> &w) {
/*
    Applies a filter on the mixture. Just corresponds to a matrix
    multiplication.

    Parameters
    x: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        STFT of the signal on which to apply the filter.

    w: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        filtering matrices, as returned, e.g. by :func:`wiener_gain`

    Returns
    y: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        filtered signal
*/
    size_t nb_frames = x.shape(0), nb_bins = x.shape(1), nb_channels = x.shape(2);

    py::array_t<complex> y({nb_frames, nb_bins, nb_channels});
    std::fill(y.mutable_data(), y.mutable_data() + y.size(), 0);

    auto x_ref = x.unchecked<3>();
    auto w_ref = w.unchecked<4>();
    auto y_ref = y.mutable_unchecked<3>();

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t i = 0; i < nb_channels; i++) {
                for (size_t j = 0; j < nb_channels; j++) {
                    complex delta = cref(w, frame, bin, j, i) * cref(x, frame, bin, i);
                    ref(y, frame, bin, j) += delta;
                }
            }
        }
    }

    return y;
}


void covariance(py::array_t<complex> &r_j, const py::array_t<complex> &y_j, size_t frame) {
/*
    Compute the empirical covariance for a source.

    Parameters:
    in/out:
        r_j: np.ndarray [shape=(nb_bins, nb_channels, nb_channels)]
             Output parameter: just y_j * conj(y_j.T): empirical covariance for each TF bin.
        y_j: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)].
             complex stft of the source.
*/
    size_t nb_bins = y_j.shape(1), nb_channels = y_j.shape(2);

    auto r_j_ref = r_j.mutable_unchecked<3>();
    auto y_j_ref = y_j.unchecked<3>();

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t i = 0; i < nb_channels; i++) {
            for (size_t j = 0; j < nb_channels; j++) {
                complex y1 = cref(y_j, frame, bin, i);
                complex y2 = cref(y_j, frame, bin, j);

                ref(r_j, bin, i, j) += y1 * std::conj(y2);
            }
        }
    }
}

void get_local_gaussian_model(py::array_t<double> &v_j, py::array_t<complex> &r_j,
                              const py::array_t<complex> &y_j, double eps) {
/*
    Compute the local Gaussian model [1]_ for a source given the complex STFT.
    First get the power spectral densities, and then the spatial covariance
    matrix, as done in [1]_, [2]_

    Parameters.
    out:
        v_j: np.ndarray [shape=(nb_frames, nb_bins)]
            power spectral density of the source
        r_j: np.ndarray [shape=(nb_bins, nb_channels, nb_channels)]
            spatial covariance matrix of the source
    in:
        y_j: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
              complex STFT of the source.
        eps: float [scalar]
            regularization term
*/
    size_t nb_frames = y_j.shape(0), nb_bins = y_j.shape(1), nb_channels = y_j.shape(2);
    std::vector<double> weight(nb_bins, eps);

    auto v_j_ref = v_j.mutable_unchecked<2>();
    auto r_j_ref = r_j.mutable_unchecked<3>();
    auto y_j_ref = y_j.unchecked<3>();

    for (size_t frame = 0; frame < nb_frames; frame++) {
        for (size_t bin = 0; bin < nb_bins; bin++) {
            double total = 0.0;

            for (size_t ch = 0; ch < nb_channels; ch++) {
                const complex &val = cref(y_j, frame, bin, ch);
                double re = std::real(val), im = std::imag(val);
                total += re * re + im * im;
            }

            ref(v_j, frame, bin) = total / nb_channels;
        }
    }

    for (size_t frame = 0; frame < nb_frames; frame++) {
        covariance(r_j, y_j, frame);
    }

    for (size_t frame = 0; frame < nb_frames; frame++) {
        for (size_t bin = 0; bin < nb_bins; bin++) {
            weight[bin] += cref(v_j, frame, bin);
        }
    }

    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t i = 0; i < nb_channels; i++) {
            for (size_t j = 0; j < nb_channels; j++) {
                ref(r_j, bin, i, j) /= weight[bin];
            }
        }
    }
}

py::array_t<complex> get_phase(py::array_t<float> &v, const py::array_t<complex> &x) {
/*
    Calculates v * np.exp(1j * np.angle(x[..., None])

    Parameters
        v: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
           Spectrograms
        x: np.ndarray [shape=(nb_frames, nb_bins, nb_sources)]
           STFT of the mixture signal.
    Returns
        v: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
           Initial approximation
*/
    size_t nb_frames = v.shape(0), nb_bins = v.shape(1), nb_channels = v.shape(2), nb_sources = v.shape(3);
    py::array_t<complex> res({nb_frames, nb_bins, nb_channels, nb_sources});

    auto res_ref = res.mutable_unchecked<4>();
    auto v_ref = v.unchecked<4>();
    auto x_ref = x.unchecked<3>();

    // #pragma omp for
    for (size_t frame = 0; frame < nb_frames; frame++) {
        for (size_t bin = 0; bin < nb_bins; bin++) {
            for (size_t ch = 0; ch < nb_channels; ch++) {
                // Use Euler's formula to calculate v * np.exp(1j * np.angle(x))
                complex x_val = cref(x, frame, bin, ch);
                double re = std::real(x_val), im = std::imag(x_val);

                double eps = 1e-20;
                double abs_re = std::abs(re), abs_im = std::abs(im);
                abs_re = std::max(abs_re, eps);

                double tan = abs_im / abs_re;
                double cos = 1 / std::sqrt(1 + tan * tan);
                double sin = tan * cos;

                if (re < 0) cos *= -1.0;
                if (im < 0) sin *= -1.0;

                complex mult = complex{cos, sin};

                for (size_t source = 0; source < nb_sources; source++) {
                    ref(res, frame, bin, ch, source) = mult * double(cref(v, frame, bin, ch, source));
                }
            }
        }
    }

    return res;
}

double downscale(py::array_t<complex> &x, py::array_t<complex> &y) {
/*
    Scales both arrays by the maximum absolute value from x.

    Parameters:
        inout:
            x: np.ndarray, 3D
            y: np.ndarray, 4D

    Returns:
        scale value
*/
    size_t nb_frames = y.shape(0), nb_bins = y.shape(1), nb_channels = y.shape(2), nb_sources = y.shape(3);

    auto x_ref = x.mutable_unchecked<3>();
    auto y_ref = y.mutable_unchecked<4>();

    double max_abs = 1.0;

    // #pragma omp for
    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t ch = 0; ch < nb_channels; ch++) {
                max_abs = std::max(max_abs, std::abs(cref(x, frame, bin, ch)) / 10);
            }
        }
    }

    double inv_max_abs = 1.0 / max_abs;

    // #pragma omp for
    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t ch = 0; ch < nb_channels; ch++) {
                ref(x, frame, bin, ch) *= inv_max_abs;

                for (size_t source = 0; source < nb_sources; source++) {
                    ref(y, frame, bin, ch, source) *= inv_max_abs;
                }
            }
        }
    }

    return max_abs;
}

void upscale(py::array_t<complex> &y, double scale) {
/*
    Scales one array by the given value.

    Parameters:
        inout:
            x: np.ndarray, 3D
        in:
            scale

    Returns:
        scale value
*/
    size_t nb_frames = y.shape(0), nb_bins = y.shape(1), nb_channels = y.shape(2), nb_sources = y.shape(3);
    auto y_ref = y.mutable_unchecked<4>();

    // #pragma omp for
    for (size_t bin = 0; bin < nb_bins; bin++) {
        for (size_t frame = 0; frame < nb_frames; frame++) {
            for (size_t ch = 0; ch < nb_channels; ch++) {
                for (size_t source = 0; source < nb_sources; source++) {
                    ref(y, frame, bin, ch, source) *= scale;
                }
            }
        }
    }
}


PYBIND11_MODULE(fast_norbert_cpp, m) {
    m.def("get_mix_model", &get_mix_model, py::arg("v_j").noconvert(), py::arg("v_j").noconvert());
    m.def("wiener_gain", &wiener_gain, py::arg("v_j").noconvert(), py::arg("r_j").noconvert(),
          py::arg("inv_cxx").noconvert());
    m.def("apply_filter", &apply_filter,py::arg("x").noconvert(), py::arg("v").noconvert());
    m.def("get_local_gaussian_model", &get_local_gaussian_model, py::arg("v_j").noconvert(),
          py::arg("r_j").noconvert(), py::arg("y_j").noconvert(), py::arg("r_j"));
    m.def("get_phase", &get_phase, py::arg("v").noconvert(), py::arg("x").noconvert());
    m.def("downscale", &downscale, py::arg("x").noconvert(), py::arg("y").noconvert());
    m.def("upscale", &upscale, py::arg("y").noconvert(), py::arg("scale"));
}
