import torch
from torch.autograd import Variable
import numpy as np
from collections import Iterable


def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad.data if x.grad is not None else None
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result


def zero_gradients(i):
    for t in iter_gradients(i):
        if t is not None:
            t.zero_()


def make_jacobian(input, num_out):
    if isinstance(input, Variable) and not input.requires_grad:
        return None
    if torch.is_tensor(input) or isinstance(input, Variable):
        return torch.zeros(input.nelement(), num_out)
    else:
        return type(input)(filter(lambda x: x is not None,
                                  (make_jacobian(elem, num_out) for elem in input)))


def iter_tensors(x, only_requiring_grad=False):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Variable):
        if x.requires_grad or not only_requiring_grad:
            yield x.data
    else:
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def contiguous(input):
    if torch.is_tensor(input):
        return input.contiguous()
    elif isinstance(input, Variable):
        return input.contiguous()
    else:
        return type(input)(contiguous(e) for e in input)


def get_numerical_jacobian(fn, input, target, eps=1e-3):
    # To be able to use .view(-1) input must be contiguous
    input = contiguous(input)
    output_elements = fn(input).numel()
    jacobian = make_jacobian(target, output_elements)
    output_size = fn(input).size()
    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    outa = torch.FloatTensor(output_size)
    outb = torch.FloatTensor(output_size)

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1)
        for i in range(flat_tensor.nelement()):
            orig = flat_tensor[i]
            flat_tensor[i] = orig - eps
            outa.copy_(fn(input))
            flat_tensor[i] = orig + eps
            outb.copy_(fn(input))
            flat_tensor[i] = orig

            grad = (outb.numpy() - outa.numpy())/ (2*eps)
            # outb.add_(-1, outa).div_(2 * eps)
            d_tensor[i] = torch.from_numpy(grad)

    return jacobian


# def get_analytical_jacobian(input, output):
#     jacobian = make_jacobian(input, output.numel())
#     grad_output = output.data.clone().zero_()
#     flat_grad_output = grad_output.view(-1)
#
#     for i in range(flat_grad_output.numel()):
#         flat_grad_output.zero_()
#         flat_grad_output[i] = 1
#         zero_gradients(input)
#         output.backward(grad_output, retain_graph=True)
#         for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
#             if d_x is None:
#                 jacobian_x[:, i].zero_()
#             else:
#                 jacobian_x[:, i] = d_x.to_dense() if d_x.is_sparse else d_x
#
#     return jacobian
def iter_variables(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield (x.grad.data, x.data) if x.grad is not None else (None, None)
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_variables(elem):
                yield result

def get_analytical_jacobian(input, output):
    input = contiguous(input)
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            zero_gradients(input)
            output.backward(grad_output, create_graph=True)
            for jacobian_x, (d_x, x) in zip(jacobian_c, iter_variables(input)):
                if jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        jacobian_x[:, i] = d_x.to_dense() if d_x.is_sparse else d_x
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() != 0:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,

# def fail_test(msg):
#         if raise_exception:
#             raise RuntimeError(msg)
#         return False
def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3):
    """Check gradients computed via small finite differences
       against analytical gradients

    The check between numerical and analytical has the same behaviour as
    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
    is true for all elements of analytical jacobian a and numerical jacobian n.

    Args:
        func: Python function that takes Variable inputs and returns
            a tuple of Variables
        inputs: tuple of Variables
        eps: perturbation for finite differences
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        True if all differences satisfy allclose condition
    """
    output = func(*inputs)
    output = _as_tuple(output)
    Flag = True
    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i].data
        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
        # print analytical
        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)
        # print numerical
        for k,  (a, n) in enumerate(zip(analytical, numerical)):
            # relative_loss = (a - n) / (n + eps)
            # print relative_loss
            if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                 max_diff = (np.abs((a-n).numpy()).max())
                 max_diff_indices = np.where(np.abs((a-n).numpy()) == max_diff)

                 print "{:d}th Input Grad is problematic, max diff:{:.06f}".format(k, max_diff)
                 for index_x, index_y in zip(max_diff_indices[0], max_diff_indices[1]):
                     print '[{:d}, {:d}]'.format(index_x, index_y)


                 Flag = False
            # else:
                # print "{:d}th Input Grad is None problematic".format(k)

        if not reentrant:
            print ('not reentrant')

        if not correct_grad_sizes:
            print ('not correct_grad_sizes')

    if False == Flag:
        print "Backward Prop Error Found!"
    # check if the backward multiplies by grad_output
    zero_gradients(inputs)
    output = _as_tuple(func(*inputs))
    torch.autograd.backward(output, [o.data.new(o.size()).zero_() for o in output])
    for input_id, i in enumerate(inputs):
        if i.grad is None:
            continue
        if not i.grad.data.eq(0).all():
            print '{:d}th input is problematic mutiplies by gradout'.format(input_id)
            Flag = False

    return Flag
