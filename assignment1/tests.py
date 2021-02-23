import unittest
import numpy as np
import student
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
class ConvolutionTestCase(unittest.TestCase):
    def setUp(self):
        self.random_img = np.random.randn(101,101)
        self.random_filt = np.random.randn(5,5)

    def testSize(self):
        output = student.convolve(self.random_img, self.random_filt)
        self.assertEqual(output.shape, (101,101), 'Incorrect output size')

    def testValues(self):
        output = student.convolve(self.random_img, self.random_filt)
        output_gt = convolve2d(self.random_img, self.random_filt, mode='same')
        self.assertTrue(np.allclose(output, output_gt), 'Incorrect values')

    def testValuesFlipped(self):
        output = student.convolve(self.random_filt, self.random_img)
        output_gt = convolve2d(self.random_filt, self.random_img, mode='same')
        self.assertTrue(np.allclose(output, output_gt), 'Incorrect values when performing convolution with a filter smaller than the image')

class GaussianTestCase(unittest.TestCase):
    def setUp(self):
        self.random_img = np.random.randn(100,100)

    def testSum(self):
        output = student.gaussian_filter(5,1)
        self.assertTrue(np.allclose(output.sum(),1),'Filters must sum to 1.')

    def testValues(self):
        filt = student.gaussian_filter(5,1)
        out = convolve2d(self.random_img, filt, mode='same')
        out_gt = gaussian_filter(self.random_img, sigma=1, truncate=2,mode='constant')
        self.assertTrue(np.allclose(out, out_gt))

class GradientTestCase(unittest.TestCase):
    def setUp(self):
        X = np.arange(0,10,1).reshape((1,-1))
        Y = np.ones((X.size,1))
        self.img = np.matmul(Y,X)
        self.const = np.ones(self.img.shape)
        self.diag = X + X.T

    def testderivX_Xramp(self):
        filt = student.dx_filter()
        output = convolve2d(self.img, filt, mode='same')
        self.assertTrue(output[5,5]>0.1, 'X derivative filter does not detect horizontal gradient')

    def testderivX_Yramp(self):
        filt = student.dx_filter()
        output = convolve2d(self.img.T, filt, mode='same')
        self.assertTrue(np.abs(output[5,5])<1e-5, 'X derivative filter fires on vertical gradient')

    def testderivX_const(self):
        filt = student.dx_filter()
        output = convolve2d(self.const, filt, mode='same')
        self.assertTrue(np.abs(output[5,5])<1e-5, 'X derivative filter fires on constant image')

    def setUp(self):
        X = np.arange(0,10,1).reshape((1,-1))
        Y = np.ones((X.size,1))
        self.img = np.matmul(Y,X)
        self.const = np.ones(self.img.shape)
        self.diag = (X+X.T)/np.sqrt(2)
        self.noisy = np.zeros((100,100))
        self.noisy[:,50:] = 1
        self.noisy = self.noisy + 0.2*np.random.randn(*self.noisy.shape)

    def testderivY_Xramp(self):
        filt = student.dy_filter()
        output = convolve2d(self.img.T, filt, mode='same')
        self.assertTrue(output[5,5]>0.1, 'Y derivative filter does not detect vertical gradient')

    def testderivY_Yramp(self):
        filt = student.dy_filter()
        output = convolve2d(self.img, filt, mode='same')
        self.assertTrue(np.abs(output[5,5])<1e-5, 'Y derivative filter fires on horizontal gradient')

    def testderivY_const(self):
        filt = student.dy_filter()
        output = convolve2d(self.const, filt, mode='same')
        self.assertTrue(np.abs(output[5,5])<1e-5, 'Y derivative filter fires on constant image')

    def testGradient_Xramp(self):
        filtx = student.dx_filter()
        output = convolve2d(self.img, filtx, mode='same')
        grad = student.gradient_magnitude(self.img)
        self.assertTrue(np.allclose(np.abs(output[5,5]), grad[5,5]), 'Gradient magnitude for horizontal gradient must match X derivative')

    def testGradient_diagonal(self):

        grad1 = student.gradient_magnitude(self.img)
        grad2 = student.gradient_magnitude(self.diag)
        self.assertTrue(np.allclose(grad1[5,5], grad2[5,5]), 'Gradient magnitude should stay the same if edge is rotated')

    def testGradient_noisy(self):
        grad1 = student.gradient_magnitude(self.noisy)
        grad2 = student.gradient_magnitude(self.noisy, k=11, sigma=2)
        ratio1 = np.max(grad1[:,10:40])/np.max(grad1[:,40:60])
        ratio2 = np.max(grad2[:,10:40])/np.max(grad2[:,40:60])

        self.assertTrue(ratio2<ratio1)

        








if __name__ == '__main__':
    unittest.main()
