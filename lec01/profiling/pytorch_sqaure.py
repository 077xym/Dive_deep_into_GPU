import torch


def time_pytorch_function(func, input):
    # CUDA is ASYNC and thus cannot use python built-in time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


b = torch.randn(10000, 10000).cuda()


def square_2(a):
    return a * a


def square_3(a):
    return a ** 2


print(time_pytorch_function(torch.square, b))
print(time_pytorch_function(square_2, b))
print(time_pytorch_function(square_3, b))

print()

print("==============")
print("profiling torch.sqaure")
print("==============")

# using torch.autograd.profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print()

print("==============")
print("profiling torch.sqaure")
print("==============")

# using torch.autograd.profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print()

print("==============")
print("profiling torch.sqaure")
print("==============")

# using torch.autograd.profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

