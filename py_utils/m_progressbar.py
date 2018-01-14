import sys
import time

def progress_bar(k,n, prefix_message='Progress', post_message='', start_time=None, newline=False):
    """
    Show the progress bar
    k: current progress
    n: total

    Created: 23-Mar-2017, Last modified: 23-Mar-2017
    """

    n_digit = len("{:d}".format(n))

    str_format = "{:s} {{:{:d}d}}/{:d} ({{:6.2f}}%), {:s}".format(prefix_message, n_digit, n, post_message)
    if start_time:
        if newline:
            str_format += " elapse time: {:7.1f}s\n".format(time.time() - start_time)

        else:
            str_format += " elapse time: {:7.1f}s".format(time.time() - start_time)

    pre_str = '\r'
    post_str = ''
    if k == 1:
        pre_str = '\n'
    elif k == n:
        post_str = '\n'

    _ = sys.stdout.write(pre_str + str_format.format(k, 100 * k / n) + post_str)
    # sys.stdout.flush()