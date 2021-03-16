def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Completed', decimals = 1, length = 100, fill = '█', printEnd = "\n"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    # filledLength = int(length * iteration // total)
    # bar = fill * filledLength + '-' * (length - filledLength)
    status = 0
    if iteration == total:
        status = 1
    if iteration == -1:
        status = -1
    # return(f"{status}\t{prefix} |{bar}| {percent}% {suffix}{printEnd}")
    return(f"{status}\t{iteration}{printEnd}")

def logd(file, mode, progress, text=''):
    with open(file, mode) as f:
        f.write(printProgressBar(progress, 100))
        # f.write(text + '\n')
    f.close()

def status_from_logd(file):
    with open(file, "r") as f:
        lines = f.readlines()
    f.close()
    return lines[-1].split('\t')
