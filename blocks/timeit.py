import time

def timeit(method):
    """ Decorator for timing execution of a method in a class """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        classname  = repr(args[0]).split(' ')[0].split('.')[-1]
        methodname = str(method.__name__)
        time_str   = '%2.2fs' % (te-ts)

        print '%32s %16s %16s' % (classname, methodname, time_str)
        return result
    return timed