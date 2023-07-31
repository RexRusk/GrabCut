import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import igraph as ig
from sklearn.mixture import GaussianMixture
import math


class GrabCutSegmentation:
    def __init__(self, image_path):
        self.img = cv.imread(image_path)
        self.img_final = None
        self.mask = None
        self.mask_final = None
        self.rect = [0, 0, 0, 0]
        self.rect_copy = [0, 0, 0, 0]
        self.BG = 0  # Background
        self.FG = 1  # Foreground
        self.PR_BG = 2  # Probable background
        self.PR_FG = 3  # Probable foreground
        self.leftButtonDown = False
        self.leftButtonUp = True
        self.leftButtonDown = None
        self.leftButtonUp = None

    '''Control your mouse to select a bounding box using your left click and release buttons'''

    def on_mouse(self, event, x, y, flag, param):

        # Pressed
        if event == cv.EVENT_LBUTTONDOWN:
            self.rect[0] = x
            self.rect[2] = x
            self.rect[1] = y
            self.rect[3] = y
            self.leftButtonDown = True
            self.leftButtonUp = False

        # Move
        if event == cv.EVENT_MOUSEMOVE:
            if self.leftButtonDown and not self.leftButtonUp:
                self.rect[2] = x
                self.rect[3] = y

        # Release
        if event == cv.EVENT_LBUTTONUP:
            if self.leftButtonDown and not self.leftButtonUp:
                x_min = min(self.rect[0], self.rect[2])
                y_min = min(self.rect[1], self.rect[3])

                x_max = max(self.rect[0], self.rect[2])
                y_max = max(self.rect[1], self.rect[3])

                self.rect[0] = x_min
                self.rect[1] = y_min
                self.rect[2] = x_max
                self.rect[3] = y_max
                self.leftButtonDown = False
                self.leftButtonUp = True

    '''
    Calculates Euclidean distance in colour space used in equation 11 to estimate the 
    foreground and background regions of an image based on color similarity and smoothness 
    constraints of alpha

    '''

    def calculateWeights(self, Zm, Zn, beta, gamma, diag=False):
        # Function to calculate weights used in the GrabCut algorithm
        weight = beta * np.sum((Zm - Zn) * (Zm - Zn), axis=2)
        weight = gamma * np.exp(-weight)
        if diag:
            weight = weight / np.sqrt(2)  # alpha n != alpha m
        return weight.flatten()  # flatten into a 1D array

    '''
        Calculate beta - parameter of GrabCut algorithm.
        First we estimate the square color difference for all neighbouring pixels
        Then use the edge formula in graph theory: edges = 2*(4*cols*rows-3*(cols+rows)+2)
        and beta = 1/(2*avg(sqr(||color[i] - color[j]||))) to calculate the beta
    '''

    def calculateBeta(self, img):
        # Function to calculate beta parameter
        rows, cols, _ = img.shape
        beta = 0
        for y in range(rows):
            for x in range(cols):
                color = img[y, x]
                if x > 0:  # left
                    diff = color - img[y, x - 1]
                    beta += np.dot(diff, diff)
                if y > 0 and x > 0:  # up left
                    diff = color - img[y - 1, x - 1]
                    beta += np.dot(diff, diff)
                if y > 0:  # up
                    diff = color - img[y - 1, x]
                    beta += np.dot(diff, diff)
                if y > 0 and x < cols - 1:  # up right
                    diff = color - img[y - 1, x + 1]
                    beta += np.dot(diff, diff)

        if beta <= np.finfo(float).eps:
            beta = 0
        else:
            # The number of the edges
            beta = 1 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
        return beta

    '''
    Calculates the smoothness term for each vertices. 
    The smoothness term  helps adjust the energy value between neighboring pixels 
    of an image when the contrast of the image is high or low
    '''

    def calculateSmoothnessTermV(self, img, gamma):
        # Function to calculate the smoothness term
        rows, cols, _ = img.shape

        # load the set of pairs of neighboring pixels C
        left = img[:, 1:]
        right = img[:, :-1]
        top = img[1:, :]
        bottom = img[:-1, :]
        lefttop = img[1:, 1:]
        leftbottom = img[:-1, 1:]
        righttop = img[1:, :-1]
        rightbottom = img[:-1, :-1]

        # Calculate beta
        beta = self.calculateBeta(img)

        # Calculate weights of noterminal vertices of graph.
        weight = self.calculateWeights(left, right, beta, gamma)
        weight = np.hstack((weight, self.calculateWeights(top, bottom, beta, gamma)))
        weight = np.hstack((weight, self.calculateWeights(lefttop, rightbottom, beta, gamma, True)))
        weight = np.hstack((weight, self.calculateWeights(leftbottom, righttop, beta, gamma, True)))
        tem = np.array([i for i in range(rows * cols)])
        idx = tem.reshape(rows, cols)
        # return weight.tolist(),edges
        hor1 = idx[:, 1:].flatten()
        hor2 = idx[:, :-1].flatten()
        # print(hor1.shape)
        hor = np.array([hor1, hor2]).T
        # up_neighbourhood
        ver1 = idx[1:, :].flatten()
        ver2 = idx[:-1, :].flatten()
        ver = np.array([ver1, ver2]).T
        # upleft_neighbourhood
        diag11 = idx[1:, 1:].flatten()
        diag12 = idx[:-1, :-1].flatten()
        diag1 = np.array([diag11, diag12]).T
        # upright_neighbourhood
        diag21 = idx[1:, 1:].flatten()
        diag22 = idx[:-1, :-1].flatten()
        diag2 = np.array([diag21, diag22]).T
        edges = np.vstack((hor, ver, diag1, diag2))

        return weight, edges

    '''
    Calculate the weight and edge of each pixel after calculate smoothness term
    sedg is source edge
    tedg is sink edge

    '''

    def calculateEdgesWeights(self, img, bgd_gmm, Vweights, Vedges, fgd_gmm, mask, s, t, gamma):
        # Function to calculate the edge weights
        weight, edgesbin = self.calculateSmoothnessTermV(img, gamma)
        # Find indices for probable and known regions
        prob_idx = np.where(np.logical_or(mask.reshape(-1) == self.PR_BG, mask.reshape(-1) == self.PR_FG))
        back_idx = np.where(mask.reshape(-1) == self.BG)
        fore_idx = np.where(mask.reshape(-1) == self.FG)
        # Create edges for the graph
        edg = Vedges

        # The loop calculates the s1 and t1 arrays for each set of indices
        indices_list = [prob_idx[0], back_idx[0], fore_idx[0]]
        for indices in indices_list:
            s1 = np.ones(indices.size) * s
            t1 = np.ones(indices.size) * t
            sedg = np.array([s1, indices]).T
            tedg = np.array([t1, indices]).T
            edg = np.vstack((edg, sedg, tedg))
        edges = [(int(x), int(y)) for x, y in edg]

        # Update weights with GMM likelihoods for probable background and foreground regions
        weights = Vweights

        # Equation 8 and 9
        # Calculates the negative log-likelihood for the probable FG/BG region of the data term
        weights = np.hstack((weights, -bgd_gmm.score_samples(img.reshape(-1, 3)[prob_idx])))
        weights = np.hstack((weights, -fgd_gmm.score_samples(img.reshape(-1, 3)[prob_idx])))

        weights = np.hstack((weights, np.zeros(back_idx[0].size)))
        weights = np.hstack((weights, np.ones(back_idx[0].size) * 9 * gamma))
        weights = np.hstack((weights, np.zeros(fore_idx[0].size)))
        weights = np.hstack((weights, np.ones(fore_idx[0].size) * 9 * gamma))
        weights = weights.tolist()

        weights.extend(weight)
        edges.extend(edgesbin)
        return edges, weights

    '''
    Use this function to show the line chart of energy convergence
    TODO Minimize the energy in iterations
    '''

    def calculateEnergy(self, mask, weights):
        # Function to calculate the energy in equation 7
        weights = np.array(weights)  # Convert the weights list to a NumPy array
        energy = np.sum(np.where(mask == 2, weights[1], 0)) + np.sum(np.where(mask == 3, weights[2], 0))
        return energy

    '''
    The algorithm is based on graph cut optimization and iteratively refines 
    the segmentation to separate the foreground and background regions of an image. 
    '''

    def grabcut(self, img, mask, bbox, gmm_components=5, num_iters=3):
        # Function to perform the GrabCut algorithm
        print("GrabCut iteration has started")

        # Initialize the parameters
        rows, cols = img.shape[0], img.shape[1]
        tem = np.array([i for i in range(rows * cols)])  # Temp value
        idx = tem.reshape(rows, cols)  # The index of each pixel of the image
        s = cols * rows  # The size of a set S
        t = s - 1  # to ensure a minimum of 2 is required in GMMs that t is always distinct from s in the GraphCut
        # img = img.astype(np.float64) # convert uint8 to float64, to be more precise
        gamma = 50  # This value was obtained from opencv cpp codes as a standard number

        # Initializes GMMs
        if np.count_nonzero(mask) == 0:
            mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 3
        bgd_gmm = GaussianMixture(n_components=gmm_components, covariance_type='full')
        fgd_gmm = GaussianMixture(n_components=gmm_components, covariance_type='full')
        Vedges, Vweights = self.calculateSmoothnessTermV(img, gamma)

        energy_list = []  # Store energy values for each iteration

        for num in range(num_iters):
            print("iteration number " + str(num) + "...")

            # 1. Assign GMM components to pixels: for each n in TU
            back_idx = np.where(np.logical_or(mask == 0, mask == 2))
            fore_idx = np.where(np.logical_or(mask == 1, mask == 3))
            # 2. Estimate model with EM algorithm
            bgd_gmm.fit(img[back_idx])
            fgd_gmm.fit(img[fore_idx])

            # FG/BG GMM prediction
            edges, weights = self.calculateEdgesWeights(img, bgd_gmm, Vedges, Vweights, fgd_gmm, mask, s, t, gamma)

            # Perform GraphCut to save to the mask
            graph = ig.Graph(2 + cols * rows)
            graph.add_edges(edges)
            pr_indexes = np.where(np.logical_or(mask == 2, mask == 3))

            # 3. Estimate segmentation: use min cut to solve
            mincut = graph.st_mincut(s, t, weights)

            # Update the mask based on the GraphCut result
            mask[pr_indexes] = np.where(np.isin(idx[pr_indexes], mincut.partition[0]), 3, 2)

            # Calculate and store the energy for this iteration
            energy = self.calculateEnergy(mask, weights)
            energy_list.append(energy)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')  # The mask of the foreground

        print("done")
        print("Press any keyboard key to continue the border matting")

        # Plot the energy convergence
        plt.plot(range(num_iters), energy_list)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Convergence')
        plt.show()

        return mask2, mask

    def run(self, num_iters=3):
        # Function to run the GrabCut segmentation
        self.mask = np.zeros(self.img.shape[:2], np.uint8)

        # Create a named window and set the mouse callback
        cv.namedWindow('img')
        cv.setMouseCallback('img', self.on_mouse)
        cv.imshow('img', self.img)

        while cv.waitKey(2) == -1:
            if self.leftButtonDown and not self.leftButtonUp:
                img_copy = self.img.copy()
                cv.rectangle(img_copy, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (0, 255, 0), 2)
                cv.imshow('img', img_copy)
            elif not self.leftButtonDown and self.leftButtonUp and self.rect[2] - self.rect[0] != 0 and self.rect[3] - \
                    self.rect[1] != 0:
                # Calculate the width and height
                self.rect[2] = self.rect[2] - self.rect[0]
                self.rect[3] = self.rect[3] - self.rect[1]
                self.rect_copy = self.rect.copy()
                self.rect = [0, 0, 0, 0]

                # TODO Make the iteration constructor to the test case
                mask2, _ = self.grabcut(self.img, self.mask, self.rect_copy, 5, num_iters)

                # img_show = img * mask2[:, :, np.newaxis]

                img_final = cv.bitwise_and(self.img, self.img, mask=mask2)

                self.img_final = img_final
                self.mask_final = mask2
                cv.imshow('GrabCutImg', self.img_final)  # Show the foreground output

                # cv.imshow('GrabCutMask', self.mask_final)  # Show the foreground mask

        cv.destroyAllWindows()


class TrimapGenerator:
    def __init__(self, image, mask, rect=()):
        self.image = image
        self.mask = mask
        self.rect = rect

    '''Crop the raw image, return a cropped bounding box image and a foreground white color cropped trimap'''

    def crop(self):
        # Resize the image and trimap to the original image size
        cropped_image = self.image[self.rect[1]:self.rect[1] + self.rect[3], self.rect[0]:self.rect[0] + self.rect[2]]
        cropped_trimap = self.mask[self.rect[1]:self.rect[1] + self.rect[3], self.rect[0]:self.rect[0] + self.rect[2]]

        # Show the cropped image
        cv.imshow("Before Border Matting", cropped_image)
        # cv.imshow("cropped Trimap", cropped_trimap)
        return cropped_image, cropped_trimap


class BorderMatting:
    def __init__(self, img, trimap):
        self.cropped_image = img  # Save the cropped image
        self.img = img
        self.trimap = trimap  # 0: background, 255: foreground
        self.w = 6  # TU is the set of pixels in a ribbon of width ±w pixels either side of C
        self.lambda1 = 50  # for smoothing regularizer
        self.lambda2 = 1000  # for smoothing regularizer
        self.L = 20  # for sampling mean and sample variance (41-1)/2
        self.delta_level = 30  # for minimizing energy function (DP)
        self.sigma_level = 10  # for minimizing energy function (DP)
        self.C = []  # contour
        self.D = dict()  # dictionary for t(n): format (xt, yt): [(x1, y1), ...]
        self.delta_sigma_dict = dict()  # dictionary for delta and sigma: format (xt, yt): (delta, sigma)

    def run(self):
        print("\nfinding contour...")  # show progress
        self.findContour()  # find contour
        print("\nfind contour distance...")  # show progress
        self.pixelGroup()  # group pixels and map them to contour pixels
        print("\nminimizing energy function...")  # show progress
        self.energyFunction()  # minimizing energy function: find delta and sigma pairs
        alpha_map = self.constructAlphaMap()  # use best delta and sigma pairs to construct alpha map
        print("\ncompleted")

        # output_pixel = (alpha_map * foreground_pixel) + ((1 - alpha_map) * background_pixel(0, 0, 0))
        alpha_map = np.array(alpha_map)
        # print(self.cropped_image)
        # print(alpha_map)
        border_matting_image = (alpha_map[:, :, np.newaxis] * self.cropped_image).astype(np.uint8)
        # print(border_matting_image)

        cv.imshow("Aftet Border Matting", border_matting_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    ''' Main Utility Functions '''

    def findContour(self):
        # TODO: To write edge finder by myself
        self.trimap = np.uint8(self.trimap)
        # Erode the edges
        self.trimap = cv.erode(self.trimap, kernel=np.ones((3, 3), np.uint8), iterations=2)
        edges = cv.Canny(self.trimap, threshold1=2, threshold2=3)

        # construct new trimap, eliminate the edge background color
        newmap = np.zeros_like(self.trimap)
        newmap[self.trimap == 0] = 0
        newmap[self.trimap == 255] = 4
        newmap[edges == 255] = 0
        self.trimap = newmap
        # print(self.trimap)

        # Find the contour
        indices = np.where(edges == 255)
        self.C = list(zip(indices[0], indices[1]))
        # print(self.C)
        return

    '''Find the near contour pixels for each pixel in the trimap'''

    def pixelGroup(self):
        for point in self.C:
            self.D[point] = []

        m, n = self.trimap.shape
        # Calculate the minimum Euclidean distance
        for i in range(m):
            for j in range(n):
                min_dist = 100000000
                min_point = None
                for point in self.C:
                    dist = (i - point[0]) ** 2 + (j - point[1]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_point = point
                if min_dist < self.w ** 2:
                    self.D[min_point].append((i, j))

        # for keys in self.D.keys():
        #     print(keys, self.D[keys])
        return

    ''' Equation 12 in the paper '''

    def energyFunction(self):

        # TODO: check time complexity
        # Previous delta and sigma
        _delta = 1
        _sigma = 1

        for point in self.C:
            energy = 10000000000000000000000000000
            best_delta = None
            best_sigma = None
            for delta in range(1, self.delta_level):
                for sigma in range(1, self.sigma_level):
                    delta = delta / self.delta_level * self.w
                    sigma = sigma / self.sigma_level * self.w
                    V = self.smoothingRegularizer(delta, _delta, sigma, _sigma)
                    D = 0
                    pixelGroup = self.D[point]
                    for pixel in pixelGroup:
                        distance = ((pixel[0] - point[0]) ** 2 + (pixel[1] - point[1]) ** 2) ** 0.5
                        if self.trimap[pixel[0]][pixel[1]] == 0:
                            distance = -distance
                        alpha = self.alphaDistance(distance, sigma, delta)
                        # print(alpha)
                        tmp = self.dataTerm(alpha, point)
                        D += tmp
                        # print(tmp)
                    if energy > V + D:
                        # print("energy: ", energy)
                        energy = V + D
                        best_delta = delta
                        best_sigma = sigma

            # Check if best_delta and best_sigma are still None (no suitable values found)
            if best_delta is None:
                best_delta = _delta
            if best_sigma is None:
                best_sigma = _sigma
            self.delta_sigma_dict[point] = (best_delta, best_sigma)
            _delta = best_delta
            _sigma = best_sigma

        # for keys in self.delta_sigma_dict.keys():
        #     print(keys, self.delta_sigma_dict[keys])
        return

    '''
    The alpha map represents the transparency or opacity of each pixel in the image. 
    The values in the alpha map range from 0 to 1, where 0 represents completely transparent 
    (fully background) and 1 represents completely opaque (fully foreground). Intermediate 
    values between 0 and 1 indicate partial transparency.
    '''

    def constructAlphaMap(self):
        # Alpha map initialization
        m, n = self.trimap.shape
        alpha_map = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if self.trimap[i][j] == 0:  # Background
                    alpha_map[i][j] = 0
                elif self.trimap[i][j] == 4:  # Foreground
                    alpha_map[i][j] = 1
                else:
                    alpha_map[i][j] = -1
        # Construct the alpha map
        for point in self.C:
            delta, sigma = self.delta_sigma_dict[point]
            pixelGroup = self.D[point]
            for pixel in pixelGroup:
                distance = ((pixel[0] - point[0]) ** 2 + (pixel[1] - point[1]) ** 2) ** 0.5
                if self.trimap[pixel[0]][pixel[1]] == 0:
                    distance = -distance
                alpha = self.alphaDistance(distance, sigma, delta)
                # print(alpha)
                alpha_map[pixel[0]][pixel[1]] = alpha
            distance = 0
            alpha = self.alphaDistance(distance, sigma, delta)
            alpha_map[point[0]][point[1]] = alpha
        # print(alpha_map)
        return alpha_map

    ''' equation 13 in the paper '''

    def smoothingRegularizer(self, delta, _delta, sigma, _sigma):

        # print(delta, _delta, sigma, _sigma)
        return self.lambda1 * (delta - _delta) ** 2 + self.lambda2 * (sigma - _sigma) ** 2

    ''' Equation 14 in the paper '''

    def dataTerm(self, alpha, pos):

        out = self.gaussian(alpha, self.alphaMean(alpha, pos), self.alphaVariance(alpha, pos)) / math.log(2)
        if out <= 0:
            return 0
        else:
            return -1 * math.log(out)

    ''' Equation 15 in the paper '''

    def alphaMean(self, alpha, pos):

        out = (1 - alpha) * self.sampleMean(pos, 0) + alpha * self.sampleMean(pos, 1)
        # print("alpha mean: ", out)
        return out

    ''' Equation 15 in the paper '''

    def alphaVariance(self, alpha, pos):

        out = (1 - alpha) ** 2 * self.sampleVariance(pos, 0) + alpha ** 2 * self.sampleVariance(pos, 1)
        return out

    ''' Sample mean and covariance in L*L sample space '''

    def sampleMean(self, pos, alpha):
        area = self.img[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        trimap = self.trimap[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        if alpha == 0:  # background
            mean = np.sum(area[trimap == 0]) / self.L ** 2
        else:  # foreground
            mean = np.sum(area[trimap == 4]) / self.L ** 2
        # print("sample mean: ", mean)
        return mean

    def sampleVariance(self, pos, alpha):
        area = self.img[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        trimap = self.trimap[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        if alpha == 0:  # background
            variance = np.sum((area[trimap == 0] - self.sampleMean(pos, alpha)) ** 2) / self.L ** 2
        else:  # foreground
            variance = np.sum((area[trimap == 4] - self.sampleMean(pos, alpha)) ** 2) / self.L ** 2
        # print("sample variance: ", variance)
        return variance

    '''Distance to alpha'''

    def alphaDistance(self, distance, sigma, delta):
        if distance < 0:
            return 0
        return 1 / (1 + np.exp(-1 * (distance - delta) / sigma))

    '''gaussian distribution formula'''

    def gaussian(self, x, mean, variance):
        epsilon = 1e-8  # A small positive value to avoid division by zero
        variance += epsilon  # Add epsilon to the variance to avoid zero or very small covariance
        out = np.exp(-(x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        # print(out)
        return out


if __name__ == "__main__":
    # Replace with the path to your image
    image_path = ".\images\elefant.jpg"

    # New GrabCut test case
    newSegmentationTestCase = GrabCutSegmentation(image_path)
    newSegmentationTestCase.run(3)  # Default iteration is 3

    # New crop
    newCrop = TrimapGenerator(newSegmentationTestCase.img_final, newSegmentationTestCase.mask_final,
                              newSegmentationTestCase.rect_copy)
    a, b = newCrop.crop()

    # New Border Matting test case
    newBorderMatting = BorderMatting(a, b)
    newBorderMatting.run()  # Take some time to minimize energy
