"""Numerical kernels for mojo-librosa's C ABI."""

from std.algorithm import parallelize
from std.math import cos, exp, floor, log, pow, sin, sqrt
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime W32 = simdwidthof[DType.float32]()
comptime PI = 3.14159265358979323846264338327950288


@export("mls_initialize")
def mls_initialize() abi("C"):
    initialize_runtime()


def fft_inplace(data: Ptr, n: Int):
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            var tr = data[2 * i]
            var ti = data[2 * i + 1]
            data[2 * i] = data[2 * j]
            data[2 * i + 1] = data[2 * j + 1]
            data[2 * j] = tr
            data[2 * j + 1] = ti

    var length = 2
    while length <= n:
        var angle = -2.0 * PI / Float64(length)
        var wlen_r = cos(angle)
        var wlen_i = sin(angle)
        var base = 0
        while base < n:
            var wr = 1.0
            var wi = 0.0
            for k in range(length >> 1):
                var even = base + k
                var odd = even + (length >> 1)
                var odd_r = data[2 * odd] * wr - data[2 * odd + 1] * wi
                var odd_i = data[2 * odd] * wi + data[2 * odd + 1] * wr
                var even_r = data[2 * even]
                var even_i = data[2 * even + 1]
                data[2 * even] = even_r + odd_r
                data[2 * even + 1] = even_i + odd_i
                data[2 * odd] = even_r - odd_r
                data[2 * odd + 1] = even_i - odd_i
                var next_wr = wr * wlen_r - wi * wlen_i
                wi = wr * wlen_i + wi * wlen_r
                wr = next_wr
            base += length
        length <<= 1


def stft_frame(
    y: Ptr,
    window: Ptr,
    dst: Ptr,
    index: Int,
    n_samples: Int,
    n_frames: Int,
    n_fft: Int,
    hop_length: Int,
):
    var channel = index // n_frames
    var frame = index - channel * n_frames
    var data = dst + 2 * index * n_fft
    var start = frame * hop_length
    for i in range(n_fft):
        data[2 * i] = y[channel * n_samples + start + i] * window[i]
        data[2 * i + 1] = 0.0
    fft_inplace(data, n_fft)


@export("mls_stft")
def mls_stft(
    y_addr: Int,
    window_addr: Int,
    dst_addr: Int,
    channels: Int,
    n_samples: Int,
    n_frames: Int,
    n_fft: Int,
    hop_length: Int,
) abi("C"):
    var y = Ptr(unsafe_from_address=y_addr)
    var window = Ptr(unsafe_from_address=window_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var count = channels * n_frames

    @parameter
    def work(index: Int):
        stft_frame(y, window, dst, index, n_samples, n_frames, n_fft, hop_length)

    if count * n_fft >= 131072:
        parallelize[work](count)
    else:
        for index in range(count):
            work(index)


def project_row(
    matrix: Ptr,
    src: Ptr,
    dst: Ptr,
    index: Int,
    rows: Int,
    inner: Int,
    columns: Int,
):
    var b = index // rows
    var r = index - b * rows
    var src_batch = src + b * inner * columns
    var dst_row = dst + (b * rows + r) * columns
    for c in range(columns):
        dst_row[c] = 0.0
    for k in range(inner):
        var weight = matrix[r * inner + k]
        var src_row = src_batch + k * columns
        var vw = SIMD[DType.float64, W](weight)
        var c = 0
        while c + W <= columns:
            dst_row.store(
                c,
                dst_row.load[width=W](c) + vw * src_row.load[width=W](c),
            )
            c += W
        while c < columns:
            dst_row[c] += weight * src_row[c]
            c += 1


def project_row_f32(
    matrix: F32Ptr,
    src: F32Ptr,
    dst: F32Ptr,
    index: Int,
    rows: Int,
    inner: Int,
    columns: Int,
):
    var b = index // rows
    var r = index - b * rows
    var src_batch = src + b * inner * columns
    var dst_row = dst + (b * rows + r) * columns
    for c in range(columns):
        dst_row[c] = 0.0
    for k in range(inner):
        var weight = matrix[r * inner + k]
        var src_row = src_batch + k * columns
        var vw = SIMD[DType.float32, W32](weight)
        var c = 0
        while c + W32 <= columns:
            dst_row.store(
                c,
                dst_row.load[width=W32](c)
                + vw * src_row.load[width=W32](c),
            )
            c += W32
        while c < columns:
            dst_row[c] += weight * src_row[c]
            c += 1


@export("mls_project")
def mls_project(
    matrix_addr: Int,
    src_addr: Int,
    dst_addr: Int,
    batch: Int,
    rows: Int,
    inner: Int,
    columns: Int,
) abi("C"):
    """Project each row-major spectrogram through a shared matrix."""
    var matrix = Ptr(unsafe_from_address=matrix_addr)
    var src = Ptr(unsafe_from_address=src_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var count = batch * rows

    @parameter
    def work(index: Int):
        project_row(matrix, src, dst, index, rows, inner, columns)

    if count >= 64 and inner * columns >= 32768:
        parallelize[work](count)
    else:
        for index in range(count):
            work(index)


@export("mls_project_f32")
def mls_project_f32(
    matrix_addr: Int,
    src_addr: Int,
    dst_addr: Int,
    batch: Int,
    rows: Int,
    inner: Int,
    columns: Int,
) abi("C"):
    var matrix = F32Ptr(unsafe_from_address=matrix_addr)
    var src = F32Ptr(unsafe_from_address=src_addr)
    var dst = F32Ptr(unsafe_from_address=dst_addr)
    var count = batch * rows

    @parameter
    def work(index: Int):
        project_row_f32(matrix, src, dst, index, rows, inner, columns)

    if count >= 64 and inner * columns >= 32768:
        parallelize[work](count)
    else:
        for index in range(count):
            work(index)


def sinc(x: Float64) -> Float64:
    if abs(x) < 1.0e-14:
        return 1.0
    return sin(PI * x) / (PI * x)


@export("mls_resample")
def mls_resample(
    y_addr: Int,
    weights_addr: Int,
    dst_addr: Int,
    channels: Int,
    n_in: Int,
    n_out: Int,
    radius: Int,
    phase_count: Int,
    input_step: Int,
) abi("C"):
    var y = Ptr(unsafe_from_address=y_addr)
    var weights = Ptr(unsafe_from_address=weights_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var kernel_width = 2 * radius + 1
    var count = channels * n_out

    @parameter
    def work_item(index: Int):
        var channel = index // n_out
        var i = index - channel * n_out
        var phase = i % phase_count
        var cycle = i // phase_count
        var phase_numerator = phase * input_step
        var base = cycle * input_step + phase_numerator // phase_count
        var left = base - radius
        var weight_row = weights + phase * kernel_width
        var acc = 0.0
        if left >= 0 and left + kernel_width <= n_in:
            var source_row = y + channel * n_in + left
            var vacc0 = SIMD[DType.float64, W](0.0)
            var vacc1 = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + 2 * W <= kernel_width:
                vacc0 += (
                    source_row.load[width=W](k)
                    * weight_row.load[width=W](k)
                )
                vacc1 += (
                    source_row.load[width=W](k + W)
                    * weight_row.load[width=W](k + W)
                )
                k += 2 * W
            while k + W <= kernel_width:
                vacc0 += (
                    source_row.load[width=W](k)
                    * weight_row.load[width=W](k)
                )
                k += W
            acc = (vacc0 + vacc1).reduce_add()
            while k < kernel_width:
                acc += source_row[k] * weight_row[k]
                k += 1
        else:
            for k in range(kernel_width):
                var source_index = left + k
                if source_index >= 0 and source_index < n_in:
                    acc += y[channel * n_in + source_index] * weight_row[k]
        dst[index] = acc

    if count * kernel_width >= 262144:
        var chunks = (count + 255) // 256

        @parameter
        def work_chunk(chunk: Int):
            var first = chunk * 256
            var last = min(first + 256, count)
            for index in range(first, last):
                work_item(index)

        parallelize[work_chunk](chunks)
    else:
        for index in range(count):
            work_item(index)


@export("mls_resample_f32")
def mls_resample_f32(
    y_addr: Int,
    weights_addr: Int,
    dst_addr: Int,
    channels: Int,
    n_in: Int,
    n_out: Int,
    radius: Int,
    phase_count: Int,
    input_step: Int,
) abi("C"):
    var y = F32Ptr(unsafe_from_address=y_addr)
    var weights = F32Ptr(unsafe_from_address=weights_addr)
    var dst = F32Ptr(unsafe_from_address=dst_addr)
    var kernel_width = 2 * radius + 1
    var count = channels * n_out

    @parameter
    def work_item(index: Int):
        var channel = index // n_out
        var i = index - channel * n_out
        var phase = i % phase_count
        var cycle = i // phase_count
        var phase_numerator = phase * input_step
        var base = cycle * input_step + phase_numerator // phase_count
        var left = base - radius
        var weight_row = weights + phase * kernel_width
        var acc: Float32 = 0.0
        if left >= 0 and left + kernel_width <= n_in:
            var source_row = y + channel * n_in + left
            var vacc0 = SIMD[DType.float32, W32](0.0)
            var vacc1 = SIMD[DType.float32, W32](0.0)
            var k = 0
            while k + 2 * W32 <= kernel_width:
                vacc0 += (
                    source_row.load[width=W32](k)
                    * weight_row.load[width=W32](k)
                )
                vacc1 += (
                    source_row.load[width=W32](k + W32)
                    * weight_row.load[width=W32](k + W32)
                )
                k += 2 * W32
            while k + W32 <= kernel_width:
                vacc0 += (
                    source_row.load[width=W32](k)
                    * weight_row.load[width=W32](k)
                )
                k += W32
            acc = (vacc0 + vacc1).reduce_add()
            while k < kernel_width:
                acc += source_row[k] * weight_row[k]
                k += 1
        else:
            for k in range(kernel_width):
                var source_index = left + k
                if source_index >= 0 and source_index < n_in:
                    acc += y[channel * n_in + source_index] * weight_row[k]
        dst[index] = acc

    if count * kernel_width >= 262144:
        var chunks = (count + 255) // 256

        @parameter
        def work_chunk(chunk: Int):
            var first = chunk * 256
            var last = min(first + 256, count)
            for index in range(first, last):
                work_item(index)

        parallelize[work_chunk](chunks)
    else:
        for index in range(count):
            work_item(index)


@export("mls_resample_slow")
def mls_resample_slow(
    y_addr: Int,
    dst_addr: Int,
    channels: Int,
    n_in: Int,
    n_out: Int,
    orig_sr: Float64,
    target_sr: Float64,
    half_width: Int,
) abi("C"):
    var y = Ptr(unsafe_from_address=y_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var cutoff = target_sr / orig_sr
    if cutoff > 1.0:
        cutoff = 1.0
    var support = Float64(half_width) / cutoff
    for channel in range(channels):
        for i in range(n_out):
            var position = Float64(i) * orig_sr / target_sr
            var left = Int(floor(position - support))
            var right = Int(floor(position + support))
            var acc = 0.0
            var weight_sum = 0.0
            for j in range(left, right + 1):
                var distance = position - Float64(j)
                var phase = distance / support
                if abs(phase) > 1.0:
                    continue
                var weight = cutoff * sinc(cutoff * distance)
                weight *= 0.5 + 0.5 * cos(PI * phase)
                if j >= 0 and j < n_in:
                    acc += y[channel * n_in + j] * weight
                weight_sum += weight
            if abs(weight_sum) > 1.0e-15:
                acc /= weight_sum
            dst[channel * n_out + i] = acc


@export("mls_beat_dp")
def mls_beat_dp(
    onset_addr: Int,
    local_addr: Int,
    cum_addr: Int,
    backlink_addr: Int,
    n: Int,
    period: Int,
    tightness: Float64,
) abi("C"):
    var onset = Ptr(unsafe_from_address=onset_addr)
    var local = Ptr(unsafe_from_address=local_addr)
    var cum = Ptr(unsafe_from_address=cum_addr)
    var backlink = IPtr(unsafe_from_address=backlink_addr)

    var peak = 0.0
    var kernel_width = 2 * period + 1
    if kernel_width <= n:
        for k in range(kernel_width):
            var z = Float64(k - period) * 32.0 / Float64(period)
            cum[k] = exp(-0.5 * z * z)

    @parameter
    def local_score(i: Int):
        var score = 0.0
        var first = i - period
        if first < 0:
            first = 0
        var last = i + period
        if last >= n:
            last = n - 1
        if kernel_width <= n:
            var kernel_offset = first - i + period
            var length = last - first + 1
            var vacc = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= length:
                vacc += (
                    onset.load[width=W](first + k)
                    * cum.load[width=W](kernel_offset + k)
                )
                k += W
            score = vacc.reduce_add()
            while k < length:
                score += onset[first + k] * cum[kernel_offset + k]
                k += 1
        else:
            for j in range(first, last + 1):
                var z = Float64(i - j) * 32.0 / Float64(period)
                score += exp(-0.5 * z * z) * onset[j]
        local[i] = score

    if n * kernel_width >= 131072:
        var chunks = (n + 255) // 256

        @parameter
        def local_chunk(chunk: Int):
            var first = chunk * 256
            var last = min(first + 256, n)
            for i in range(first, last):
                local_score(i)

        parallelize[local_chunk](chunks)
    else:
        for i in range(n):
            local_score(i)

    for i in range(n):
        if local[i] > peak:
            peak = local[i]

    var min_distance = Int(floor(Float64(period) * 0.5 + 0.5))
    var max_distance = min(2 * period, n - 1)
    for distance in range(min_distance, max_distance + 1):
        var deviation = log(Float64(distance)) - log(Float64(period))
        onset[distance] = -tightness * deviation * deviation

    var threshold = 0.01 * peak
    var first_beat = True
    for i in range(n):
        var best_score = -1.7976931348623157e308
        var beat_location = -1
        var loc = i - min_distance
        var end = i - 2 * period
        while loc >= end:
            if loc < 0:
                break
            var score = cum[loc] + onset[i - loc]
            if score > best_score:
                best_score = score
                beat_location = loc
            loc -= 1
        if beat_location >= 0:
            cum[i] = local[i] + best_score
        else:
            cum[i] = local[i]
        if first_beat and local[i] < threshold:
            backlink[i] = -1
        else:
            backlink[i] = Int64(beat_location)
            first_beat = False


@export("mls_tempo_period")
def mls_tempo_period(
    onset_addr: Int,
    n: Int,
    min_period: Int,
    max_period: Int,
    start_period: Float64,
) abi("C") -> Int:
    var onset = Ptr(unsafe_from_address=onset_addr)
    var best_period = min_period
    var best_score = -1.7976931348623157e308
    var upper = max_period
    if upper >= n:
        upper = n - 1
    for period in range(min_period, upper + 1):
        var corr = 0.0
        for i in range(period, n):
            corr += onset[i] * onset[i - period]
        var prior = log(Float64(period) / start_period) / log(2.0)
        var score = log(1.0 + 1.0e6 * max(corr, 0.0)) - 0.5 * prior * prior
        if score > best_score:
            best_score = score
            best_period = period
    return best_period
