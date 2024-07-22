import numpy as np
import time


test_list = [-1, 0, 1, 2, 3, 5, 7, 999]

thresholds = np.load("MultiThreshold_0_param0.npy")
thresholds = np.reshape(thresholds, thresholds.shape[0]*thresholds.shape[1])

print(thresholds[0])
print(thresholds[255])
print(thresholds[254])
print(thresholds[509])


def upper_bound(li, val):
    count = len(li)
    first = 0

    while (count > 0):
        indexptr = first
        step = count // 2
        indexptr += step

        if (not (val < li[indexptr])):
            indexptr += 1
            first = indexptr
            count -= step + 1
        else:
            count = step

    return first


def multithreshold(elemcount, inp):
    ret = [-128] * len(inp)
    if (elemcount == len(inp)):
        # case batchsize 1
        for elemindex in range(elemcount):
            ret[elemindex] += upper_bound(thresholds[elemindex *
                                          255:(elemindex+1) * 255], inp[elemindex])
    else:
        # case all other batch sizes
        for elemindex in range(elemcount):
            last = -float("inf")
            indexLast = 0
            for batchindex in range(len(inp)//elemcount):
                curr = inp[batchindex * elemcount + elemindex]
                indexCurr = 0
                if (curr == last):
                    indexCurr = indexLast
                elif (curr > last):
                    indexCurr = upper_bound(
                        thresholds[elemindex * 255 + indexLast:(elemindex+1) * 255], curr)
                    indexCurr += indexLast
                else:
                    indexCurr = upper_bound(
                        thresholds[elemindex * 255:(elemindex+1) * 255 - (255 - indexLast)], curr)

                ret[batchindex * elemcount + elemindex] += indexCurr
                last = curr
                indexLast = indexCurr

    return ret


"""constinit float a = 255 / (thresholds[254] - thresholds[0]);

    std::vector<int8_t> multithresholdLinearPerTensor(const std::vector<float>& inp) {
        const size_t size = inp.size();
        std::vector<int8_t> ret(size, -128);
        std::vector<int> protoRet(size);
        std::size_t threadcount = std::min({ 24ul ,static_cast<std::size_t>(omp_get_num_procs()), FinnUtils::fastLog2(inp.size() >> 4) });
        omp_set_num_threads(threadcount);
#pragma omp for simd
        for (size_t i = 0; i < size; ++i) {
            protoRet[i] = std::clamp(static_cast<int>((inp[i] - thresholds[0]) * a), 0, 254);
        }
#pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            const int val = protoRet[i];
            ret[i] += static_cast<int>(inp[i] - thresholds[val] + 1.0f) + val;
        }
        return ret;
    }"""

a = 255 / (thresholds[254] - thresholds[0])


def clamp(val, ul, ll):
    temp = val + ul - abs(val - ul)
    if ll == 0:
        return int((temp + abs(temp))*0.25)
    else:
        lowerTimes2 = 2*ll
        return int((temp + lowerTimes2 + abs(temp - lowerTimes2)) * 0.25)


def multithresholdLinearPerTensor(inp):
    size = len(inp)
    ret = [-128] * size
    for i in range(size):
        temp = clamp(int((inp[i] - thresholds[0]) * a), 254, 0)
        ret[i] += int(inp[i] - thresholds[temp] + 1.0) + temp
    return ret


testinputs = [0.5527185, 0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -
              0.09802981,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323, 0.27337435,  -0.11517189,  0.37859723, -0.15901394, 0.29185423, -0.344608, 0.08293352]

print(multithreshold(24, testinputs))

testinputs = testinputs * 4

print(multithreshold(24, testinputs))

# print(upper_bound(test_list, 0))
# print(upper_bound(test_list, 2))
# print(upper_bound(test_list, 66))

inp = [-0.021009455, 0.019932786, -0.039395217, 0.18287413, 0.00012183236, -0.032901216, 0.25017667, -0.022611154, 0.00185829, 0.020921344, 0.012650512, -0.04234076, -0.051785916, -0.061475027, -0.004451474, 0.62080276, -0.014915439, 0.029105086, 0.7079885, 0.026286222, 0.0018582224, -0.13297302, -0.007866903, 0.037671357, 0.64609253, -0.010580023, -0.045337804, -0.13296913, -0.032439955, -0.021977415, -0.1267499, -0.057640415, -0.05359094, 0.17047033, -0.017160123, -
       0.06444771, 1.4546201, -0.029384447, -0.052251257, -0.161837, 0.076085255, 0.28753442, -0.111458495, -0.04988761, -0.053051833, -0.07829765, 0.0317371, -0.06444789]
expectedOut = [27.0, 0.0, -2.0, -6.0, -1.0, -1.0, -5.0, -2.0, -2.0, 7.0, -
               1.0, -3.0, 60.0, -1.0, -2.0, -7.0, 3.0, 12.0, -5.0, -2.0, -2.0, -3.0, 1.0, -3.0]

print(multithreshold(24, inp))
print(multithresholdLinearPerTensor(inp))


randomInputs = (4 - -4) * np.random.random((4096*24)) - 4
print(type(randomInputs))
# print(multithreshold(24, randomInputs))

timecounter = 0
for t in range(1000):
    start = time.perf_counter()
    multithreshold(24, randomInputs)
    stop = time.perf_counter()
    timecounter += stop - start

print(timecounter/1000)

randomInputs = (4 - -4) * np.random.random((4096*24)) - 4
print(type(randomInputs))
# print(multithreshold(24, randomInputs))

timecounter = 0
for t in range(1000):
    start = time.perf_counter()
    multithresholdLinearPerTensor(randomInputs)
    stop = time.perf_counter()
    timecounter += stop - start

print(timecounter/1000)
