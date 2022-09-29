/* chebyshev.c

A Python C extension module for fitting multiple exponential and harmonic
functions using Chebyshev polynomials.

Refer to the chebyshev.py module for documentation and tests.

For theoretical background see
Analytic solutions to modelling exponential and harmonic functions using
Chebyshev polynomials: fitting frequency-domain lifetime images with
photobleaching. G C Malachowski, R M Clegg, and G I Redford.
J Microsc. 2007; 228(3): 282-295. doi: 10.1111/j.1365-2818.2007.01846.x

:Authors:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_,
  Oliver Holub

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2015.03.19

Requirements
------------
*  `CPython 2.7 or 3.4 <http://www.python.org>`_
*  `Numpy 1.9 <http://www.numpy.org>`_

Install
-------
Use this Python distutils setup script to build the extension module::

  # setup.py
  # Usage: ``python setup.py build_ext --inplace``
  from distutils.core import setup, Extension
  import numpy
  setup(name='_chebyshev',
        ext_modules=[Extension('_chebyshev', ['chebyshev.c'],
                               include_dirs=[numpy.get_include()])])

License
-------
Copyright (c) 2008-2015, Christoph Gohlke
Copyright (c) 2008-2015, The Regents of the University of California
Produced at the Laboratory for Fluorescence Dynamics.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holders nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#define _VERSION_ "2015.03.19"

#define WIN32_LEAN_AND_MEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "math.h"
#include "float.h"
#include "string.h"
#include "numpy/arrayobject.h"

#define MAXCOEF 64  /* maximum number of polynomial coefficients */
#define MAXEXPS 8   /* maximum number of exponential components */
#define PIVOT_TOLERANCE 1.0e-14
#define LAGUERRE_TOLERANCE 1.0e-12
#define TWOPI 6.283185307179586476925286766559
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/*****************************************************************************/
/* C functions */

/*
Find single root of complex polynomial using Laguerre's method.
*/
int laguerre(
    int numpoly,
    Py_complex* coeff,
    Py_complex* root)
{
    Py_complex p, dp, ddp, gg, g, h, f, dx, gpf, gmf;

    int i, j;
    int n = numpoly - 1;
    double t, u;

    root->real = 0.5;
    root->imag = 0.0;

    for (i = 0; i < 30; i++) {
        /* evaluate polynomial */
        p.real = coeff[n].real;
        p.imag = coeff[n].imag;
        dp.real = dp.imag = ddp.real = ddp.imag = 0.0;
        for (j = 1; j < numpoly; j++) {
            t = ddp.real;
            ddp.real = t*root->real - ddp.imag*root->imag + 2.0*dp.real;
            ddp.imag = ddp.imag*root->real + t*root->imag + 2.0*dp.imag;
            t = dp.real;
            dp.real = t*root->real - dp.imag*root->imag + p.real;
            dp.imag = dp.imag*root->real + t*root->imag + p.imag;
            t = p.real;
            p.real = t*root->real - p.imag*root->imag + coeff[n-j].real;
            p.imag = p.imag*root->real + t*root->imag + coeff[n-j].imag;
        }

        t = p.real*p.real + p.imag*p.imag;
        if (sqrt(t) < LAGUERRE_TOLERANCE)
            return 0;

        g.real = (dp.real*p.real + dp.imag*p.imag) / t;
        g.imag = (dp.imag*p.real - dp.real*p.imag) / t;

        gg.real = g.real*g.real - g.imag*g.imag;
        gg.imag = g.real*g.imag * 2.0;

        h.real = (n * (gg.real - (ddp.real*p.real + ddp.imag*p.imag) / t)
                     - gg.real) * (n-1);
        h.imag = (n * (gg.imag - (ddp.imag*p.real - ddp.real*p.imag) / t)
                     - gg.imag) * (n-1);

        t = sqrt(h.real*h.real + h.imag*h.imag);
        f.real = sqrt((t + h.real) / 2.0);
        f.imag = sqrt((t - h.real) / 2.0);
        if (h.imag < 0.0)
            f.imag = -f.imag;

        gpf.real = g.real + f.real;
        gpf.imag = g.imag + f.imag;
        t = gpf.real*gpf.real + gpf.imag*gpf.imag;

        gmf.real = g.real - f.real;
        gmf.imag = g.imag - f.imag;
        u = gmf.real*gmf.real + gmf.imag*gmf.imag;

        if (t > u) {
            dx.real = (n * gpf.real) / t;
            dx.imag = (n * -gpf.imag) / t;
        } else {
            dx.real = (n * gmf.real) / u;
            dx.imag = (n * -gmf.imag) / u;
        }

        root->real -= dx.real;
        root->imag -= dx.imag;

        t = sqrt(root->real*root->real + root->imag*root->imag);
        if (sqrt(dx.real*dx.real + dx.imag*dx.imag) <
            LAGUERRE_TOLERANCE * fmax(t, 1.0)) {
            return 0;
        }
    }
    return -1;
}

/*
Find complex roots of complex polynomial using Laguerre's method.
Coefficients are ordered from smallest to largest power and are altered.
*/
int polyroots(
    int numpoly,
    Py_complex* coeff,
    Py_complex* roots)
{
    int i, j;
    int n = numpoly;
    Py_complex tc, tt;
    Py_complex *root;

    for (i = 0; i < numpoly-1; i++) {
        root = &roots[i];

        /* find single root */
        laguerre(n, coeff, root);

        if (fabs(root->imag) < LAGUERRE_TOLERANCE)
            root->imag = 0.0;

        /* deflate */
        n--;
        tc = coeff[n-1];
        coeff[n-1] = coeff[n];
        for (j = n-2; j >= 0; j--) {
            tt = tc;
            tc = coeff[j];
            coeff[j].real = root->real * coeff[j+1].real -
                            root->imag * coeff[j+1].imag +
                            tt.real;
            coeff[j].imag = root->real * coeff[j+1].imag +
                            root->imag * coeff[j+1].real +
                            tt.imag;
        }
    }

    return 0;
}

/*
Solve linear equation system A*x = b for x using Gaussian elimination
with pivoting.
Matrix A is of shape (size, size) and is altered.
Vector b is of shape (size) and will contain solution vector x.
*/
int linsolve(
    int size,
    double *matrix,
    double *vector)
{
    double temp;
    int i, j, k, m, ks, ms, ksk, js;

    /* forward solution */
    for (k = 0; k < size-1; k++) {
        ks = k*size;
        ksk = ks + k;

        /* find maximum coefficient in column */
        m = k;
        temp = fabs(matrix[ks + k]);
        for (i = k+1; i < size; i++) {
            if (temp < fabs(matrix[i*size + k])) {
                temp = matrix[i*size + k];
                m = i;
            }
        }

        /* permutate lines with index k and m */
        if (m != k) {
            ms = m*size;
            for (i = k; i < size; i++) {
                temp = matrix[ks + i];
                matrix[ks + i] = matrix[ms + i];
                matrix[ms + i] = temp;
            }
            temp = vector[k];
            vector[k] = vector[m];
            vector[m] = temp;
        }

        /* test for singular matrix */
        if (fabs(matrix[ksk]) < PIVOT_TOLERANCE)
            return -1;

        /* triangulation of matrix with coefficients */
        for (j = k+1; j < size; j++) {
            js = j * size;
            temp = - matrix[js + k] / matrix[ksk];
            for(i = k; i < size; i++) {
                matrix[js + i] += temp * matrix[ks + i];
            }
            vector[j] += temp * vector[k];
        }
    }

    /* Backward substitution */
    for (k = size-1; k >= 0; k--) {
        ks = k*size;
        for (i = k+1; i < size; i++) {
            vector[k] -= matrix[ks + i] * vector[i];
        }
        vector[k] /= matrix[ks + k];
    }
    return 0;
}

/*
Chebyshev polynomials Tj(t) / Rj.
*/
int chebypoly(
    int numdata,  /* size of data array t */
    int numcoef,  /* number of polynomial coefficients j */
    double *poly, /* output array of polynomials of shape (numcoef, numdata) */
    int norm)     /* normalize coefficients */
{
    double buffer[2*MAXCOEF];
    double *ppoly;
    double *a;
    double nf, nm2t, aj, tj, tj1, tj2;
    int t, j, ni;

    if ((numcoef < 1) || (numdata < 1) ||
        (numcoef > numdata) || (numcoef >= MAXCOEF))
        return -1;

    ni = numdata - 1;
    nf = 2.0 / (double)ni;

    a = buffer;
    for (j = 2; j < numcoef; j++) {
        aj = (double)(j * (numdata - j));
        *a++ = (double)(2*j - 1) / aj;
        *a++ = (double)((j - 1) * (ni + j)) / aj;
    }

    for (t = 0; t < numdata; t++) {
        nm2t = (double)(ni - 2*t);
        ppoly = poly + t;
        *ppoly = tj2 = 1.0;
        ppoly += numdata;
        *ppoly = tj1 = 1.0 - nf * (double)t;
        a = buffer;
        for (j = 2; j < numcoef; j++) {
            ppoly += numdata;
            *ppoly = tj = a[0]*tj1*nm2t - a[1]*tj2;
            tj2 = tj1;
            tj1 = tj;
            a += 2;
        }
    }

    if (norm != 0) {
        ppoly = poly;
        nf = (double)numdata;
        for (j = 0; j < numcoef; j++) {
            for (t = 0; t < numdata; t++) {
                *ppoly++ /= nf;
            }
            t = 2*j;
            nf *= (double)((ni + j + 2) * (t + 1)) /
                  (double)((t + 3) * (ni - j));
        }
    }
    return 0;
}

/*
Chebyshev polynomial normalization factors Rj.
*/
int chebynorm(
    int numdata,  /* size of data array */
    int numcoef,  /* number of polynomials */
    double *norm) /* output array of shape (numcoef, ) */
{
    int j, t, n;
    double f;

    if ((numcoef < 1) || (numdata < 1) ||
        (numcoef > numdata) || (numcoef >= MAXCOEF))
        return -1;

    n = numdata - 1;
    *norm++ = f = (double)numdata;
    for (j = 0; j < numcoef-1; j++) {
        t = 2*j;
        f *= (double)((n + j + 2) * (t + 1)) / (double)((t + 3) * (n - j));
        *norm++ = f;
    }
    return 0;
}

/*
Forward Chebyshev transform dj.
*/
int chebyfwd(
    char *data,      /* array of doubles to be transformed */
    int data_stride, /* byte stride of data array; 8 if contiguous */
    int numdata,     /* size of data array */
    double *coef,    /* output array of polynomial coefficients */
    int numcoef)     /* number of polynomials used */
{
    double buffer[2*MAXCOEF];
    double *a, *pcoef;
    double nf, nm2t, ft, aj, tj, tj1, tj2;
    int t, j, ni;

    if ((numcoef < 1) || (numdata < 1) ||
        (numcoef > numdata) || (numcoef >= MAXCOEF))
        return -1;

    ni = numdata - 1;
    nf = 2.0 / (double)ni;
    a = buffer;
    for (j = 2; j < numcoef; j++) {
        aj = (double)(j * (numdata - j));
        *a++ = (double)(2*j - 1) / aj;
        *a++ = (double)((j - 1) * (ni + j)) / aj;
    }

    memset(coef, 0, numcoef * sizeof(double));
    for (t = 0; t < numdata; t++) {
        nm2t = (double)(ni - 2*t);
        tj1 = 1.0 - nf * (double)t;
        tj2 = 1.0;
        ft = *((double *)data);
        data += data_stride;
        pcoef = coef;
        *pcoef++ += ft;
        *pcoef++ += ft * tj1;
        a = buffer;
        for (j = 2; j < numcoef; j++) {
            tj = a[0]*tj1*nm2t - a[1]*tj2;
            *pcoef++ += ft * tj;
            tj2 = tj1;
            tj1 = tj;
            a += 2;
        }
    }

    ft = (double)numdata;
    pcoef = coef;
    *pcoef++ /= ft;
    for (j = 0; j < numcoef-1; j++) {
        t = 2*j;
        ft *= (double)((ni + j + 2) * (t + 1)) /
              (double)((t + 3) * (ni - j));
        *pcoef++ /= ft;
    }
    return 0;
}

/*
Inverse discrete Chebyshev transform.
*/
int chebyinv(
    double *coef,
    int numcoef,
    char *data,
    int data_stride,
    int numdata)
{
    double buffer[2*MAXCOEF];
    double *a, *pcoef;
    double nf, nm2t, ft, aj, tj, tj1, tj2;
    int t, j, ni;

    if ((numcoef < 1) || (numdata < 1) ||
        (numcoef > numdata) || (numcoef >= MAXCOEF))
        return -1;

    ni = numdata - 1;
    nf = 2.0 / (double)ni;
    a = buffer;
    for (j = 2; j < numcoef; j++) {
        aj = (double)(j * (numdata - j));
        *a++ = (double)(2*j - 1) / aj;
        *a++ = (double)((j - 1) * (ni + j)) / aj;
    }

    for (t = 0; t < numdata; t++) {
        nm2t = (double)(ni - 2*t);
        tj1 = 1.0 - nf * (double)t;
        tj2 = 1.0;
        pcoef = coef;
        ft = *pcoef++;
        ft += (*pcoef++) * tj1;
        a = buffer;
        for (j = 2; j < numcoef; j++) {
            tj = a[0]*tj1*nm2t - a[1]*tj2;
            ft += tj * (*pcoef++);
            tj2 = tj1;
            tj1 = tj;
            a += 2;
        }
        *(double*)data = ft;
        data += data_stride;
    }
    return 0;
}

/*
Fit multiple exponential function.
*/
int fitexps(
    char *data,      /* data array of doubles */
    int data_stride, /* number of bytes to move from one data value to next */
    int numdata,     /* number of double values in data array */
    double *poly,    /* precalculated normalized Chebyshev polynomial Tj(t) */
    double *coef,    /* buffer for dnj of shape (numexps+1, numcoef+1) */
    int numcoef,     /* number of coefficients */
    int numexps,     /* number of exponentials to fit */
    double deltat,   /* duration between data points */
    int startcoef,   /* start coefficient. usually equals numexps-1 */
    double *buff,    /* working buffer of shape (numexps, numdata) */
    double *result,  /* buffer to receive fitted parameters
                        offset, amp[numexps], tau[numexps], frq[numexps] */
    char *fitt,      /* buffer to receive fitted data in double [numpoints] */
    int fitt_stride) /* number bytes to move from one fitted value to next */
{
    PyThreadState *_save = NULL;
    Py_complex xroots[MAXEXPS];
    Py_complex xcoefs[MAXEXPS+1];
    double matrix[MAXEXPS*MAXEXPS];
    double vector[MAXEXPS];
    double *pbuff;
    double *ppoly;
    double *pcoef;
    double *pmat;
    double *pvec;
    double *prow;
    double *pcol;
    double *pdn0;
    double *pdn1;
    double *off = result;
    double *amp = result + 1;
    double *rat = result + 1 + numexps;
    double *frq = result + 1 + numexps + numexps;
    double sum, temp, frqn, ratn;
    int j, t, n, N, row, col, error;
    int stride = numcoef + 1;

    /* discrete Chebyshev coefficients dj */
    ppoly = poly;
    pcoef = coef;
    if (data_stride == sizeof(double)) {
        double *pdata;
        for (j = 0; j < numcoef; j++) {
            pdata = (double *)data;
            sum = 0.0;
            for (t = 0; t < numdata; t++) {
                sum += (*pdata++) * (*ppoly++);
            }
            *pcoef++ = sum;
        }
    } else {
        char *pdata;
        for (j = 0; j < numcoef; j++) {
            pdata = data;
            sum = 0.0;
            for (t = 0; t < numdata; t++) {
                sum += (*((double *) pdata)) * (*ppoly++);
                pdata += data_stride;
            }
            *pcoef++ = sum;
        }
    }

    _save = PyEval_SaveThread();

    /* integral coefficients dnj */
    N = numdata - 1;
    pdn0 = coef;
    pdn1 = coef + stride;
    for (n = 0; n < numexps; n++) {
        pdn0[numcoef] = 0.0;
        pdn1[0] = 0.0;
        for (j = 1; j < numcoef; j++) {
            pdn1[j] = ((N + j + 2) * pdn0[j + 1] / (2*j + 3) - pdn0[j] -
                       (N - j + 1) * pdn0[j - 1] / (2*j - 1)) / 2.0;
        }
        pdn0 += stride;
        pdn1 += stride;
    }

    /* regression matrix */
    pmat = matrix;
    pcol = coef;
    for (col = 0; col < numexps; col++) {
        pcol += stride;
        prow = coef;
        for (row = 0; row < numexps; row++) {
            prow += stride;
            sum = 0.0;
            for (j = startcoef; j < numcoef; j++) {
                sum += prow[j] * pcol[j];
            }
            *pmat++ = sum;
        }
    }

    /* regression vector */
    pvec = vector;
    pcoef = coef;
    for (row = 0; row < numexps; row++) {
        pcoef += stride;
        sum = 0.0;
        for (j = startcoef; j < numcoef; j++) {
            sum += coef[j] * pcoef[j];
        }
        *pvec++ = sum;
    }

    /* solve linear equation system */
    error = linsolve(numexps, matrix, vector);
    if (error != 0) {
        PyEval_RestoreThread(_save);
        return error;
    }

    /* roots of polynomial */
    for (n = 0; n < numexps; n++) {
        xcoefs[n].real = -vector[numexps - n - 1];
        xcoefs[n].imag = 0.0;
    }
    xcoefs[numexps].real = 1.0;
    xcoefs[numexps].imag = 0.0;
    error = polyroots(numexps+1, xcoefs, xroots);
    if (error != 0) {
        PyEval_RestoreThread(_save);
        return error;
    }

    /* decay rate and frequency of harmonics */
    for (n = 0; n < numexps; n++) {
        temp = xroots[n].real + 1.0;
        rat[n] = -log(temp*temp + xroots[n].imag*xroots[n].imag) / 2.0;
        frq[n] = atan2(xroots[n].imag, temp);
    }

    /* fitting amplitudes */
    /* Chebyshev transform signal for each exponential component */
    pcoef = coef + (numcoef + 1);
    pbuff = buff;
    for (n = 0; n < numexps; n++) {
        frqn = frq[n];
        ratn = rat[n];
        if (frqn == 0.0) {
            *pbuff++ = 1.0;
            for (t = 1; t < numdata; t++) {
                *pbuff++ = exp(-ratn*t);
            }
        } else if (frqn > 0) {
            *pbuff++ = 1.0;
            for (t = 1; t < numdata; t++) {
                *pbuff++ = exp(-ratn*t) * cos(frqn*t);
            }
        } else {
            *pbuff++ = 0.0;
            for (t = 1; t < numdata; t++) {
                *pbuff++ = -exp(-ratn*t) * sin(frqn*t);
            }
        }
        rat[n] = deltat / ratn;
        frq[n] /= deltat;

        /* forward Chebyshev transform */
        ppoly = poly;
        for (j = 0; j < numcoef; j++) {
            pbuff -= numdata;
            sum = 0.0;
            for (t = 0; t < numdata; t++) {
                sum += (*pbuff++) * (*ppoly++);
            }
            *pcoef++ = sum;
        }
        pcoef++;
    }

    /* regression matrix for fitting amplitudes */
    pmat = matrix;
    pcol = coef;
    for (col = 0; col < numexps; col++) {
        pcol += stride;
        prow = coef;
        for (row = 0; row < numexps; row++) {
            prow += stride;
            sum = 0.0;
            for (j = 1; j < numcoef; j++) {
                sum += prow[j] * pcol[j];
            }
            *pmat++ = sum;
        }
    }

    /* regression vector for fitting amplitudes */
    pvec = amp;
    pcoef = coef;
    for (row = 0; row < numexps; row++) {
        pcoef += stride;
        sum = 0.0;
        for (j = 1; j < numcoef; j++) {
            sum += coef[j] * pcoef[j];
        }
        *pvec++ = sum;
    }

    /* solve linear equation system for amplitudes */
    error = linsolve(numexps, matrix, amp);
    if (error != 0) {
        PyEval_RestoreThread(_save);
        return error;
    }

    /* calculate offset from zero Chebyshev coefficients */
    pcoef = coef + stride;
    temp = *coef;
    for (n = 0; n < numexps; n++) {
        temp -= amp[n] * (*pcoef);
        pcoef += stride;
    }
    *off = temp;

    /* calculate fitted data */
    if (fitt != NULL) {
        if (fitt_stride == sizeof(double)) {
            double *pfitt = (double *)fitt;
            temp = *off;
            for (t = 0; t < numdata; t++) {
                *pfitt++ = temp;
            }
            pbuff = buff;
            for (n = 0; n < numexps; n++) {
                pfitt = (double *)fitt;
                temp = amp[n];
                for (t = 0; t < numdata; t++) {
                    *pfitt++ += temp * (*pbuff++);
                }
            }
        } else {
            char *pfitt = fitt;
            temp = *off;
            for (t = 0; t < numdata; t++) {
                *(double *)pfitt = temp;
                pfitt += fitt_stride;
            }
            pbuff = buff;
            for (n = 0; n < numexps; n++) {
                pfitt = fitt;
                temp = amp[n];
                for (t = 0; t < numdata; t++) {
                    *(double *)pfitt += temp * (*pbuff++);
                    pfitt += fitt_stride;
                }
            }
        }
    }

    PyEval_RestoreThread(_save);
    return 0;
}

/*
Fit frequency-domain data with photobleaching.
*/
int fitexpsin(
    char *data,      /* data array of doubles */
    int data_stride, /* number of bytes to move from one data value to next */
    int numdata,     /* number of double values in data array */
    double *poly,    /* precalculated normalized Chebyshev polynomial Tj(t) */
    double *coef,    /* buffer for dnj of shape (numexps+1, numcoef+1) */
    int numcoef,     /* number of coefficients */
    double deltat,   /* duration between data points */
    int startcoef,   /* start coefficient. usually equals numexps-1 */
    double *buff,    /* working buffer of shape (numexps, numdata) */
    double *result,  /* buffer to receive fitted parameters
                        offset, tau, frq, amp[numexps] */
    char *fitt,      /* buffer to receive fitted data in double [numpoints] */
    int fitt_stride) /* number bytes to move from one fitted value to next */
{
    PyThreadState *_save = NULL;
    Py_complex xroots[5];
    Py_complex xcoefs[6];
    double matrix[3*3];
    double *pbuff;
    double *ppoly;
    double *pcoef;
    double *pmat;
    double *pvec;
    double *prow;
    double *pcol;
    double *pdn0;
    double *pdn1;
    double *off = result;
    double *rat = result + 1;
    double *amp = result + 2;
    double sum, temp, ratn, frqn;
    double cosw, cos2w, t0, t1, t2, t3, t4, t5, t6;
    int i, j, k, t, n, N, row, col, error;
    int stride = numcoef + 1;

    /* discrete Chebyshev coefficients dj */
    ppoly = poly;
    pcoef = coef;
    if (data_stride == sizeof(double)) {
        double *pdata;
        for (j = 0; j < numcoef; j++) {
            pdata = (double *)data;
            sum = 0.0;
            for (t = 0; t < numdata; t++) {
                sum += (*pdata++) * (*ppoly++);
            }
            *pcoef++ = sum;
        }
    } else {
        char *pdata;
        for (j = 0; j < numcoef; j++) {
            pdata = data;
            sum = 0.0;
            for (t = 0; t < numdata; t++) {
                sum += (*((double *) pdata)) * (*ppoly++);
                pdata += data_stride;
            }
            *pcoef++ = sum;
        }
    }

    _save = PyEval_SaveThread();

    /* integral coefficients dnj */
    N = numdata - 1;
    pdn0 = coef;
    pdn1 = coef + stride;
    for (n = 0; n < 3; n++) {
        pdn0[numcoef] = 0.0;
        pdn1[0] = 0.0;
        for (j = 1; j < numcoef; j++) {
            pdn1[j] = ((N + j + 2) * pdn0[j + 1] / (2*j + 3) - pdn0[j] -
                       (N - j + 1) * pdn0[j - 1] / (2*j - 1)) / 2.0;
        }
        pdn0 += stride;
        pdn1 += stride;
    }

    /* cubic polynomial coefficients a + bx + cx^2 + dx^3 */
    frqn = TWOPI / numdata;
    cosw = cos(frqn);
    cos2w = cos(frqn * 2.0);
    t0 = 1.0 / (1.0 + 2.0*cosw);
    t1 = t0 * t0 * t0;
    t2 = 9.0 * t0 - 3.0;
    t3 = (16.0*cosw - 4.0*cosw*cosw + 8.0*cos2w - 8.0*cosw*cos2w - 12.0) * t1;
    t4 = 2.0 - 6.0 * t0;
    t5 = (14.0 - 14.0*cosw - 8.0*cos2w + 4.0*cosw*cosw + 4.0*cosw*cos2w) * t1;
    t6 = (4.0*cosw + 2.0*cos2w - 6.0) * t1;
    pbuff = buff;
    for (j = startcoef; j < (numcoef - 3); j++) {
        *pbuff++ = coef[j] + coef[j+stride*2] * t2 + coef[j+stride*3] * t3;
        *pbuff++ = coef[j+stride] + coef[j+stride*2] * t4 +
                   coef[j+stride*3] * t5;
        *pbuff++ = coef[j+stride*2] * t0 + coef[j+stride*3] * t6;
        *pbuff++ = coef[j+stride*3] * t1;
    }

    /* regression coefficients */
    /* quintic polynomial (a + bx + cx^2 + dx^3)*(b + 2cx + 3dx^2) */
    memset(xcoefs, 0, 6 * sizeof(Py_complex));
    pbuff = buff;
    for (j = startcoef; j < (numcoef - 3); j++) {
        xcoefs[0].real += pbuff[0]*pbuff[1];
        xcoefs[1].real += pbuff[0]*pbuff[2]*2.0 + pbuff[1]*pbuff[1];
        xcoefs[2].real += pbuff[1]*pbuff[2]*3.0 + pbuff[0]*pbuff[3]*3.0;
        xcoefs[3].real += pbuff[2]*pbuff[2]*2.0 + pbuff[1]*pbuff[3]*4.0;
        xcoefs[4].real += pbuff[2]*pbuff[3]*5.0;
        xcoefs[5].real += pbuff[3]*pbuff[3]*3.0;
        pbuff += 4;
    }

    /* roots of quintic polynomial */
    error = polyroots(6, xcoefs, xroots);
    if (error != 0) {
        PyEval_RestoreThread(_save);
        return error;
    }

    /* decay rate and frequency of harmonics */
    /* find smallest chi-square */
    t0 = DBL_MAX;
    for (i = 0; i < 5; i++) {
        pbuff = buff;
        t1 = xroots[i].real;
        t2 = t1*t1;
        t3 = t1*t1*t1;
        sum = 0.0;
        for (j = 0; j < (numcoef - startcoef - 3); j++) {
            temp = pbuff[0] + t1*pbuff[1] + t2*pbuff[2] + t3*pbuff[3];
            sum += temp*temp;
            pbuff += 4;
        }
        if (sum < t0) {
            t0 = sum;
            k = i;
        }
    }

    *rat = ratn = log((3.0 - xroots[k].real) / (2.0*cosw + 1.0));

    /* fitting amplitudes */
    /* Chebyshev transform signal for each exponential component */
    pcoef = coef + stride;

    /* decay component */
    buff[0] = 1.0;
    for (t = 1; t < numdata; t++)
        buff[t] = exp(ratn*(double)t);
    ppoly = poly;
    for (j = 0; j < numcoef; j++) {
        sum = 0.0;
        for (t = 0; t < numdata; t++)
            sum += buff[t] * (*ppoly++);
        *pcoef++ = sum;
    }
    pcoef++;

    /* sine component */
    pbuff = buff + numdata;
    pbuff[0] = 0.0;
    for (t = 1; t < numdata; t++)
        pbuff[t] = -buff[t] * sin(frqn*(double)t);
    ppoly = poly;
    for (j = 0; j < numcoef; j++) {
        sum = 0.0;
        for (t = 0; t < numdata; t++)
            sum += pbuff[t] * (*ppoly++);
        *pcoef++ = sum;
    }
    pcoef++;

    /* cosine component */
    pbuff += numdata;
    pbuff[0] = 1.0;
    for (t = 1; t < numdata; t++)
        pbuff[t] = buff[t] * cos(frqn*(double)t);
    ppoly = poly;
    for (j = 0; j < numcoef; j++) {
        sum = 0.0;
        for (t = 0; t < numdata; t++)
            sum += pbuff[t] * (*ppoly++);
        *pcoef++ = sum;
    }
    pcoef++;

    *rat = -deltat / *rat;

    /* regression matrix for fitting amplitudes */
    pmat = matrix;
    pcol = coef;
    for (col = 0; col < 3; col++) {
        pcol += stride;
        prow = coef;
        for (row = 0; row < 3; row++) {
            prow += stride;
            sum = 0.0;
            for (j = 1; j < numcoef; j++)
                sum += prow[j] * pcol[j];
            *pmat++ = sum;
        }
    }

    /* regression vector for fitting amplitudes */
    pvec = amp;
    pcoef = coef;
    for (row = 0; row < 3; row++) {
        pcoef += stride;
        sum = 0.0;
        for (j = 1; j < numcoef; j++)
            sum += coef[j] * pcoef[j];
        *pvec++ = sum;
    }

    /* solve linear equation system for amplitudes */
    error = linsolve(3, matrix, amp);
    if (error != 0) {
        PyEval_RestoreThread(_save);
        return error;
    }

    /* calculate offset from zero Chebyshev coefficients */
    pcoef = coef + stride;
    temp = *coef;
    for (n = 0; n < 3; n++) {
        temp -= amp[n] * (*pcoef);
        pcoef += stride;
    }
    *off = temp;

    /* calculate fitted data */
    if (fitt != NULL) {
        if (fitt_stride == sizeof(double)) {
            double *pfitt = (double *)fitt;
            temp = *off;
            for (t = 0; t < numdata; t++)
                *pfitt++ = temp;
            pbuff = buff;
            for (n = 0; n < 3; n++) {
                pfitt = (double *)fitt;
                temp = amp[n];
                for (t = 0; t < numdata; t++)
                    *pfitt++ += temp * (*pbuff++);
            }
        } else {
            char *pfitt = fitt;
            temp = *off;
            for (t = 0; t < numdata; t++) {
                *(double *)pfitt = temp;
                pfitt += fitt_stride;
            }
            pbuff = buff;
            for (n = 0; n < 3; n++) {
                pfitt = fitt;
                temp = amp[n];
                for (t = 0; t < numdata; t++) {
                    *(double *)pfitt += temp * (*pbuff++);
                    pfitt += fitt_stride;
                }
            }
        }
    }

    PyEval_RestoreThread(_save);
    return 0;
}

/*****************************************************************************/
/* Python functions */

/*
Numpy array converters for use with PyArg_Parse functions.
*/

static int
PyConverter_ComplexArrayCopy(
    PyObject *object,
    PyObject **address)
{
    *address = PyArray_FROM_OTF(object, NPY_COMPLEX128,
                                NPY_ARRAY_ENSURECOPY|NPY_ARRAY_IN_ARRAY);
     if (*address == NULL) return NPY_FAIL;
     return NPY_SUCCEED;
}

static int
PyConverter_AnyDoubleArray(
    PyObject *object,
    PyObject **address)
{
    PyArrayObject *obj = (PyArrayObject *)object;
    if (PyArray_Check(object) && (PyArray_TYPE(obj) == NPY_DOUBLE)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    } else {
        *address = PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_ALIGNED);
        if (*address == NULL) {
            PyErr_Format(PyExc_ValueError, "can not convert to array");
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}

/*
Python wrapper for fitexps().
*/
char py_fitexps_doc[] =
    "Return fitted parameters and data for multi-exponential function.";

static PyObject* py_fitexps(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *fitt = NULL;
    PyArrayObject *rslt = NULL;
    PyArrayIterObject *data_it = NULL;
    PyArrayIterObject *fitt_it = NULL;
    PyArrayIterObject *rslt_it = NULL;
    Py_ssize_t newshape[NPY_MAXDIMS];
    double *poly = NULL;
    double *coef = NULL;
    double *buff = NULL;
    int i, j, error, lastaxis, numexps, numdata;
    int startcoef = -1;
    int numcoef = MAXCOEF;
    int axis = NPY_MAXDIMS;
    double deltat = 1.0;
    static char *kwlist[] = {"data", "numexps", "numcoef",
                             "deltat", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&i|idO&", kwlist,
        PyConverter_AnyDoubleArray, &data,
        &numexps, &numcoef, &deltat,
        PyArray_AxisConverter, &axis)) return NULL;

    if (axis < 0) {
        axis += PyArray_NDIM(data);
    }
    if ((axis < 0) || (axis >= NPY_MAXDIMS)) {
        PyErr_Format(PyExc_ValueError, "invalid axis");
        goto _fail;
    }
    lastaxis = PyArray_NDIM(data) - 1;

    if ((numexps < 1) || (numexps > MAXEXPS)) {
        PyErr_Format(PyExc_ValueError, "numexps out of bounds");
        goto _fail;
    }

    if ((numcoef < 1) || (numcoef > MAXCOEF)) {
        PyErr_Format(PyExc_ValueError, "numcoef out of bounds");
        goto _fail;
    }

    if (startcoef < 0) { /* start regression away from zero coefficients */
        startcoef = numexps + 1;
    }
    if (startcoef > numcoef - 2) {
        PyErr_Format(PyExc_ValueError, "startcoef out of bounds");
        goto _fail;
    }

    numdata = (int)PyArray_DIM(data, axis);
    if (numcoef > numdata)
        numcoef = numdata;

    if ((numcoef - startcoef - 1) < numexps) {
        PyErr_Format(PyExc_ValueError,
            "number of coefficients insufficient to fit data");
        goto _fail;
    }

    /* fitted data */
    fitt = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(data),
                                              PyArray_DIMS(data), NPY_DOUBLE);
    if (fitt == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate fitt array");
        goto _fail;
    }

    /* fitted parameters */
    j = 0;
    for (i = 0; i < PyArray_NDIM(data); i++) {
        if (i != axis)
            newshape[j++] = PyArray_DIM(data, i);
    }
    newshape[j] = 1 + 3*numexps;
    rslt = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(data),
                                              newshape, NPY_DOUBLE);
    if (rslt == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate rslt array");
        goto _fail;
    }

    /* working buffer */
    buff = (double *)PyMem_Malloc(numexps*numdata * sizeof(double));
    if (buff == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate buff array");
        goto _fail;
    }

    /* buffer for differential coefficients */
    coef = (double *)PyMem_Malloc((numexps+1)*(numcoef+1) * sizeof(double));
    if (coef == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate coef array");
        goto _fail;
    }

    /* precalculate normalized Chebyshev polynomial */
    poly = (double *)PyMem_Malloc(numdata * (numcoef+1) * sizeof(double));
    if (poly == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate poly");
        goto _fail;
    }

    error = chebypoly(numdata, numcoef, poly, 1);
    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebypoly() failed with error code %i", error);
        goto _fail;
    }

    /* iterate over all but specified axis */
    data_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)data, &axis);
    fitt_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)fitt, &axis);
    rslt_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)rslt, &lastaxis);

    while (data_it->index < data_it->size) {
        error = fitexps(
            (char *)data_it->dataptr,
            (int)PyArray_STRIDE(data, axis),
            numdata,
            poly,
            coef,
            numcoef,
            numexps,
            deltat,
            startcoef,
            buff,
            (double *)rslt_it->dataptr,
            (char *)fitt_it->dataptr,
            (int)PyArray_STRIDE(fitt, axis));

        if (error != 0) {
            PyErr_Format(PyExc_ValueError,
                "fitexps() failed with error code %i", error);
            goto _fail;
        }

        PyArray_ITER_NEXT(data_it);
        PyArray_ITER_NEXT(fitt_it);
        PyArray_ITER_NEXT(rslt_it);
    }

    Py_XDECREF(data_it);
    Py_XDECREF(fitt_it);
    Py_XDECREF(rslt_it);
    Py_XDECREF(data);
    PyMem_Free(poly);
    PyMem_Free(coef);
    PyMem_Free(buff);

    return Py_BuildValue("(N, N)", rslt, fitt);

  _fail:
    Py_XDECREF(data_it);
    Py_XDECREF(fitt_it);
    Py_XDECREF(rslt_it);
    Py_XDECREF(data);
    Py_XDECREF(fitt);
    Py_XDECREF(rslt);
    PyMem_Free(poly);
    PyMem_Free(coef);
    PyMem_Free(buff);

    return NULL;
}

/*
Python wrapper for fitexpsin().
*/
char py_fitexpsin_doc[] =
    "Return fit parameters and data of frequency-domain data"
    " with photobleaching.";

static PyObject* py_fitexpsin(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *fitt = NULL;
    PyArrayObject *rslt = NULL;
    PyArrayIterObject *data_it = NULL;
    PyArrayIterObject *fitt_it = NULL;
    PyArrayIterObject *rslt_it = NULL;
    Py_ssize_t newshape[NPY_MAXDIMS];
    double *poly = NULL;
    double *coef = NULL;
    double *buff = NULL;
    int i, j, error, lastaxis, numdata;
    int startcoef = -1;
    int numcoef = MAXCOEF;
    int axis = NPY_MAXDIMS;
    double deltat = 1.0;
    static char *kwlist[] = {"data", "numcoef",
                             "deltat", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|idO&", kwlist,
        PyConverter_AnyDoubleArray, &data, &numcoef, &deltat,
        PyArray_AxisConverter, &axis)) return NULL;

    if (axis < 0) {
        axis += PyArray_NDIM(data);
    }
    if ((axis < 0) || (axis >= NPY_MAXDIMS)) {
        PyErr_Format(PyExc_ValueError, "invalid axis");
        goto _fail;
    }
    lastaxis = PyArray_NDIM(data) - 1;

    if ((numcoef < 1) || (numcoef > MAXCOEF)) {
        PyErr_Format(PyExc_ValueError, "numcoef out of bounds");
        goto _fail;
    }

    if (startcoef < 0) { /* start regression away from zero coefficients */
        startcoef = 4;
    }
    if (startcoef > numcoef - 2) {
        PyErr_Format(PyExc_ValueError, "startcoef out of bounds");
        goto _fail;
    }

    numdata = (int)PyArray_DIM(data, axis);
    if (numcoef > numdata)
        numcoef = numdata;

    if ((numcoef - startcoef - 1) < 3) {
        PyErr_Format(PyExc_ValueError,
            "number of coefficients insufficient to fit data");
        goto _fail;
    }

    /* fitted data */
    fitt = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(data),
                                              PyArray_DIMS(data), NPY_DOUBLE);
    if (fitt == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate fitt array");
        goto _fail;
    }

    /* fitted parameters */
    j = 0;
    for (i = 0; i < PyArray_NDIM(data); i++) {
        if (i != axis)
            newshape[j++] = PyArray_DIM(data, i);
    }
    newshape[j] = 5;
    rslt = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(data),
                                              newshape, NPY_DOUBLE);
    if (rslt == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate rslt array");
        goto _fail;
    }

    /* working buffer */
    buff = (double *)PyMem_Malloc(3*numdata * sizeof(double));
    if (buff == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate buff array");
        goto _fail;
    }

    /* buffer for differential coefficients */
    coef = (double *)PyMem_Malloc((3+1)*(numcoef+1) * sizeof(double));
    if (coef == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate coef array");
        goto _fail;
    }

    /* precalculate normalized Chebyshev polynomial */
    poly = (double *)PyMem_Malloc(numdata * (numcoef+1) * sizeof(double));
    if (poly == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate poly");
        goto _fail;
    }

    error = chebypoly(numdata, numcoef, poly, 1);
    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebypoly() failed with error code %i", error);
        goto _fail;
    }

    /* iterate over all but specified axis */
    data_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)data, &axis);
    fitt_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)fitt, &axis);
    rslt_it = (PyArrayIterObject *)PyArray_IterAllButAxis(
                                            (PyObject *)rslt, &lastaxis);

    while (data_it->index < data_it->size) {
        error = fitexpsin(
            (char *)data_it->dataptr,
            (int)PyArray_STRIDE(data, axis),
            numdata,
            poly,
            coef,
            numcoef,
            deltat,
            startcoef,
            buff,
            (double *)rslt_it->dataptr,
            (char *)fitt_it->dataptr,
            (int)PyArray_STRIDE(fitt, axis));

        if (error != 0) {
            PyErr_Format(PyExc_ValueError,
                "fitexpsin() failed with error code %i", error);
            goto _fail;
        }

        PyArray_ITER_NEXT(data_it);
        PyArray_ITER_NEXT(fitt_it);
        PyArray_ITER_NEXT(rslt_it);
    }

    Py_XDECREF(data_it);
    Py_XDECREF(fitt_it);
    Py_XDECREF(rslt_it);
    Py_XDECREF(data);
    PyMem_Free(poly);
    PyMem_Free(coef);
    PyMem_Free(buff);

    return Py_BuildValue("(N, N)", rslt, fitt);

  _fail:
    Py_XDECREF(data_it);
    Py_XDECREF(fitt_it);
    Py_XDECREF(rslt_it);
    Py_XDECREF(data);
    Py_XDECREF(fitt);
    Py_XDECREF(rslt);
    PyMem_Free(poly);
    PyMem_Free(coef);
    PyMem_Free(buff);

    return NULL;
}

/*
Python wrapper for chebyfwd().
*/
char py_chebyfwd_doc[] =
    "Return coefficients dj of forward Chebyshev transform from data.";

static PyObject* py_chebyfwd(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *coef = NULL;
    int numdata, error;
    int numcoef = MAXCOEF;
    npy_intp t;
    static char *kwlist[] = {"data", "numcoef", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
        PyConverter_AnyDoubleArray, &data, &numcoef)) return NULL;

    if (PyArray_NDIM(data) != 1) {
        PyErr_Format(PyExc_ValueError, "not a one dimensional array");
        goto _fail;
    }

    numdata = (int)PyArray_DIM(data, 0);

    if (numcoef > numdata)
        numcoef = numdata;
    if (numcoef > MAXCOEF)
        numcoef = MAXCOEF;

    t = numcoef;
    coef = (PyArrayObject *)PyArray_SimpleNew(1, &t, NPY_DOUBLE);
    if (coef == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate coef array");
        goto _fail;
    }

    error = chebyfwd(
        (char *)PyArray_DATA(data),
        (int)PyArray_STRIDE(data, 0),
        numdata,
        (double *)PyArray_DATA(coef),
        numcoef);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebyfwd() failed with error code %i", error);
        goto _fail;
    }

    Py_DECREF(data);
    return PyArray_Return(coef);

  _fail:
    Py_XDECREF(data);
    Py_XDECREF(coef);
    return NULL;
}

/*
Python wrapper for chebyinv().
*/
char py_chebyinv_doc[] =
    "Return reconstructed data from Chebyshev coefficients dj.";

static PyObject* py_chebyinv(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *coef = NULL;
    int numcoef, error;
    int numdata = -1;
    npy_intp t;
    static char *kwlist[] = {"coef", "numdata", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
        PyConverter_AnyDoubleArray, &coef, &numdata))
        return NULL;

    if (PyArray_NDIM(coef) != 1) {
        PyErr_Format(PyExc_ValueError, "not a one dimensional array");
        goto _fail;
    }

    numcoef = (int)PyArray_DIM(coef, 0);
    if (numcoef > MAXCOEF) {
        PyErr_Format(PyExc_ValueError, "too many coefficients");
        goto _fail;
    }

    t = numdata;
    data = (PyArrayObject *)PyArray_SimpleNew(1, &t, NPY_DOUBLE);
    if (data == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate data array");
        goto _fail;
    }

    error = chebyinv(
        (double *)PyArray_DATA(coef), numcoef,
        PyArray_DATA(data), (int)PyArray_STRIDE(data, 0), numdata);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebyinv() failed with error code %i", error);
        goto _fail;
    }

    Py_DECREF(coef);
    return PyArray_Return(data);

  _fail:
    Py_XDECREF(data);
    Py_XDECREF(coef);
    return NULL;
}

/*
Python wrapper for chebypoly().
*/
char py_chebypoly_doc[] =
    "Return Chebyshev polynomials Tj(t) / Rj.";

static PyObject* py_chebypoly(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyObject *boolobj;
    PyArrayObject *poly = NULL;
    int numcoef, numdata, error;
    int norm = 0;
    Py_ssize_t shape[2];
    static char *kwlist[] = {"numdata", "numcoef", "norm", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|O", kwlist,
        &numdata, &numcoef, &boolobj))
        return NULL;

    if (boolobj != NULL)
        norm = PyObject_IsTrue(boolobj);

    if ((numcoef < 1) || (numcoef > MAXCOEF)) {
        PyErr_Format(PyExc_ValueError, "numcoef out of range");
        goto _fail;
    }

    if (numdata < 1) {
        PyErr_Format(PyExc_ValueError, "data size out of range");
        goto _fail;
    }

    if (numcoef > numdata) {
        PyErr_Format(PyExc_ValueError, "numdata < numcoef");
        goto _fail;
    }

    shape[0] = numcoef;
    shape[1] = numdata;
    poly = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    if (poly == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate poly array");
        goto _fail;
    }

    error = chebypoly(numdata, numcoef, (double *)PyArray_DATA(poly), norm);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebypoly() failed with error code %i", error);
        goto _fail;
    }

    return PyArray_Return(poly);

  _fail:
    Py_XDECREF(poly);
    return NULL;
}

/*
Python wrapper for chebynorm().
*/
char py_chebynorm_doc[] =
    "Return Chebyshev polynomial normalization factors Rj";

static PyObject* py_chebynorm(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *norm = NULL;
    int numcoef, numdata, error;
    npy_intp t;
    static char *kwlist[] = {"numdata", "numcoef", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
        &numdata, &numcoef))
        return NULL;

    if ((numcoef < 1) || (numcoef > MAXCOEF)) {
        PyErr_Format(PyExc_ValueError, "numcoef out of range");
        goto _fail;
    }

    if (numdata < 1) {
        PyErr_Format(PyExc_ValueError, "data size out of range");
        goto _fail;
    }

    if (numcoef > numdata) {
        PyErr_Format(PyExc_ValueError, "numcoef > numdata");
        goto _fail;
    }

    t = numcoef;
    norm = (PyArrayObject *)PyArray_SimpleNew(1, &t, NPY_DOUBLE);
    if (norm == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to allocate array");
        goto _fail;
    }

    error = chebynorm(numdata, numcoef, (double *)PyArray_DATA(norm));

    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "chebynorm() failed with error code %i", error);
        goto _fail;
    }

    return PyArray_Return(norm);

  _fail:
    Py_XDECREF(norm);
    return NULL;
}

/*
Python wrapper for polyroot().
*/
char py_polyroots_doc[] =
    "Return complex roots of complex polynomial using Laguerre's method.\n\n"
    "    coeffs : numpy array\n"
    "        Complex polynomial coefficients ordered from\n"
    "        smallest to largest power.\n\n";

static PyObject* py_polyroots(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyArrayObject *coeffs = NULL;
    PyArrayObject *result = NULL;
    Py_ssize_t dims;
    int error;
    static char *kwlist[] = {"coeffs", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        PyConverter_ComplexArrayCopy, &coeffs))
        return NULL;

    if (PyArray_NDIM(coeffs) != 1) {
        PyErr_Format(PyExc_ValueError, "invalid coefficients");
        goto _fail;
    }

    dims = PyArray_DIM(coeffs, 0) - 1;
    result = (PyArrayObject *)PyArray_SimpleNew(1, &dims, NPY_COMPLEX128);
    if (result == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to allocate roots array");
        goto _fail;
    }

    error = polyroots((int)PyArray_DIM(coeffs, 0), PyArray_DATA(coeffs),
                      PyArray_DATA(result));
    if (error != 0) {
        PyErr_Format(PyExc_ValueError,
            "polyroots() failed with error code %i", error);
        goto _fail;
    }

    Py_DECREF(coeffs);
    return PyArray_Return(result);

  _fail:
    Py_XDECREF(coeffs);
    Py_XDECREF(result);
    return NULL;
}

/*****************************************************************************/
/* Python module */

char module_doc[] =
    "Fit exponential and harmonic functions using Chebyshev polynomials.\n\n"
    "Refer to the associated chebyshev.py module for documentation and tests."
    "\n";

static PyMethodDef module_methods[] = {
    {"fitexps", (PyCFunction)py_fitexps,
        METH_VARARGS|METH_KEYWORDS, py_fitexps_doc},
    {"fitexpsin", (PyCFunction)py_fitexpsin,
        METH_VARARGS|METH_KEYWORDS, py_fitexpsin_doc},
    {"forward_transform", (PyCFunction)py_chebyfwd,
        METH_VARARGS|METH_KEYWORDS, py_chebyfwd_doc},
    {"inverse_transform", (PyCFunction)py_chebyinv,
        METH_VARARGS|METH_KEYWORDS, py_chebyinv_doc},
    {"polynomials", (PyCFunction)py_chebypoly,
        METH_VARARGS|METH_KEYWORDS, py_chebypoly_doc},
    {"normalization_factors", (PyCFunction)py_chebynorm,
        METH_VARARGS|METH_KEYWORDS, py_chebynorm_doc},
    {"polynomial_roots", (PyCFunction)py_polyroots,
        METH_VARARGS|METH_KEYWORDS, py_polyroots_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_chebyshev",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        module_traverse,
        module_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit__chebyshev(void)

#else

#define INITERROR return

PyMODINIT_FUNC
init_chebyshev(void)
#endif
{
    PyObject *module;

    char *doc = (char *)PyMem_Malloc(sizeof(module_doc) + sizeof(_VERSION_));
    PyOS_snprintf(doc, sizeof(module_doc) + sizeof(_VERSION_),
                  module_doc, _VERSION_);

#if PY_MAJOR_VERSION >= 3
    moduledef.m_doc = doc;
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3("_chebyshev", module_methods, doc);
#endif

    PyMem_Free(doc);

    if (module == NULL)
        INITERROR;

    if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    {
#if PY_MAJOR_VERSION < 3
    PyObject *s = PyString_FromString(_VERSION_);
#else
    PyObject *s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
