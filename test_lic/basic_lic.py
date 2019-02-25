from PIL import Image
import numpy as np
import pdb

SQUARE_FLOW_FIELD_SZ = 256
DISCRETE_FILTER_SIZE = 2048
LOWPASS_FILTER_LENGTH = 10.0
LINE_SQUARE_CLIP_MAX = 1e5
VECTOR_COMPONENT_MIN = 0.05


def SyntheszSaddle(n_xres, n_yres):
    vectr = np.zeros((n_xres, n_yres, 2))
    eps = 1e-7
    size = vectr.shape[0]
    for y in range(size):
        for x in range(size):
            xx = float(x - size/2)
            yy = float(y - size/2)
            rsq = xx**2+yy**2
            if (rsq == 0):
                vectr[y, x, 0] = -1
                vectr[y, x, 1] = 1
            else:
                vectr[y, x, 0] = -yy/rsq if yy!=0 else eps
                vectr[y, x, 1] = xx/rsq  if xx!=0 else eps
    return vectr

def NormalizVectrs(vectr):
    vectr = vectr.transpose(2, 0, 1)
    vectr = vectr / np.sqrt(np.square(vectr).sum(0))
    vectr = vectr.transpose(1, 2, 0)
    return vectr

def GenBoxFilterLUT():
    LUT_0 = np.zeros(DISCRETE_FILTER_SIZE)
    LUT_1 = np.zeros(DISCRETE_FILTER_SIZE)
    len = LUT_0.size
    for i in range(len):
        LUT_0[i] = i
        LUT_1[i] = i
    return LUT_0, LUT_1

def FlowImageLIC(n_xres, n_yres, vectr, noise, LUT_0, LUT_1, krnlen):
    ADVCTS = int(krnlen * 3) # MAXIMUM number of advection steps per direction to break dead loops
    len2ID = (DISCRETE_FILTER_SIZE - 1) / krnlen
    image = np.zeros((n_yres, n_xres))

    # for each pixle in the 2D output LIC image
    for j in range(n_yres):
        for i in range(n_xres):
            # init the composite texture accumulators and the weight accumulators
            t_acum = np.zeros(2)  # two ACcUMulated composite Textures for the two directions, perspectively
            w_acum = np.zeros(2)  # two ACcUMulated Weighting values   for the two directions, perspectively

            # for either advection direction
            for advDir in range(2):
                # init the step counter, curve-length measurer, and streamline seed
                advcts = 0  # number of ADVeCTion stepS per direction (a step counter)
                curLen = 0.0  # CURrent   LENgth of the streamline
                clp0_x = i + 0.5  # x-coordinate of CLiP point 0 (current)
                clp0_y = j + 0.5  # y-coordinate of CLiP point 0 (current)

                # access the target filter LUT
                wgtLUT = LUT_0 if advDir == 0 else LUT_1  # WeiGhT Look Up Table pointing to the target filter LUT

                # until the streamline is advected long enough or a tightly  spiralling center / focus is encountered
                while curLen < krnlen and advcts < ADVCTS:
                    # access the vector at the sample
                    vctr_x = vectr[int(clp0_y), int(clp0_x), 0]
                    vctr_y = vectr[int(clp0_y), int(clp0_x), 1]

                    # in case of a critical point
                    if vctr_x == 0 and vctr_y == 0:
                        t_acum[advDir] = 0.0 if advcts == 0 else t_acum[advDir]
                        w_acum[advDir] = 1.0 if advcts == 0 else w_acum[advDir]
                        break

                    # negate the vector for the backward-advection case
                    vctr_x = vctr_x if advDir == 0 else -vctr_x
                    vctr_y = vctr_y if advDir == 0 else -vctr_y

                    # clip the segment against the pixel boundaries -- find the shorter from the tow clipped segments
                    # replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed
                    segLen = LINE_SQUARE_CLIP_MAX
                    segLen = (int(clp0_x)-clp0_x) / vctr_x if vctr_x < -VECTOR_COMPONENT_MIN else segLen
                    segLen = (int(clp0_x)+1-clp0_x) / vctr_x if vctr_x > VECTOR_COMPONENT_MIN else segLen
                    tmpLen = (int(clp0_y)-clp0_y) / vctr_y  # TeMPorary LENgth of a trial clipped-segment
                    tmpLen = tmpLen if tmpLen < segLen else segLen
                    segLen = tmpLen if vctr_y < -VECTOR_COMPONENT_MIN else segLen
                    tmpLen = (int(clp0_y)+1-clp0_y) / vctr_y
                    tmpLen = tmpLen if tmpLen < segLen else segLen
                    segLen = tmpLen if vctr_y > VECTOR_COMPONENT_MIN else segLen

                    # update the curve-length measurers
                    prvLen = curLen  # PReVious  LENgth of the streamline
                    curLen += segLen
                    segLen += 0.0004

                    # check if the filter has reached either end
                    if curLen > krnlen:
                        curLen = krnlen
                        segLen = curLen - prvLen

                    # obtain the next clip point
                    clp1_x = clp0_x + vctr_x * segLen
                    clp1_y = clp0_y + vctr_y * segLen

                    # obtain the middle point of the segment as the texture-contributing sample
                    samp_x = (clp0_x + clp1_x) * 0.5
                    samp_y = (clp0_y + clp1_y) * 0.5

                    # obtain the texture value of the sample
                    texVal = noise[int(samp_y) % n_yres, int(samp_x) % n_xres]
                    if j == 64 and i == 64:
                        print(j, i, advcts, int(samp_y) % n_yres, int(samp_x) % n_xres)

                    # update the accumulated weight and the accumulated composite texture (texture x weight)
                    W_ACUM = wgtLUT[int(curLen * len2ID)]  # ACcuMulated Weight from the seed to the current streamline forefront
                    smpWgt = W_ACUM - w_acum[advDir]  # WeiGhT of the current SaMPle
                    w_acum[advDir] = W_ACUM
                    t_acum[advDir] += texVal * smpWgt

                    # update the step counter and the "current" clip point
                    advcts += 1
                    clp0_x = clp1_x
                    clp0_y = clp1_y

                    # check if the streamline has gone beyond the flow field
                    if clp0_x < 0.0 or clp0_x >= n_xres or clp0_y < 0.0 or clp0_y >= n_yres:
                        break

            # normalize the accumulated composite texture
            texVal = (t_acum[0] + t_acum[1]) / (w_acum[0] + w_acum[1])

            # clamp the texture value against the displayable intensity range[0, 255]
            texVal = 0.0 if texVal < 0.0 else texVal
            texVal = 255.0 if texVal > 255.0 else texVal
            image[j, i] = texVal

    image = Image.fromarray(image.astype('uint8'))
    image.save("res.jpg")

def main():
    n_xres = SQUARE_FLOW_FIELD_SZ
    n_yres = SQUARE_FLOW_FIELD_SZ

    noise = np.random.rand(n_xres, n_yres) * 80 + 100
    vectr = SyntheszSaddle(n_xres, n_yres)
    vectr = NormalizVectrs(vectr)
    LUT_0, LUT_1 = GenBoxFilterLUT()
    FlowImageLIC(n_xres, n_yres, vectr, noise, LUT_0, LUT_1, LOWPASS_FILTER_LENGTH)

if __name__ == "__main__":
    main()