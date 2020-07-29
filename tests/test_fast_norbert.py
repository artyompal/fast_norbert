
import pytest
import numpy as np

import norbert
import fast_norbert


def test_fast_norbert():
    testcase0 = np.load('tests/testcase0.npz')
    x1, v1 = testcase0['x'], testcase0['v']
    x2, v2 = np.copy(x1), np.copy(v1)

    niter = 1
    use_softmask = False

    y1 = fast_norbert.wiener(v1, x1, niter, use_softmask=use_softmask)
    y2 = norbert.wiener(v2, x2, niter, use_softmask=use_softmask)

    assert y1.shape == y2.shape, f'{y1.shape} == {y2.shape}'
    assert np.allclose(y1, y2), f'{y1.flatten()} == {y2.flatten()}'
