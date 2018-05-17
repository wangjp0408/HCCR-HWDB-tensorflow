import signal

# Based on an original idea by https://gist.github.com/nonZero/2907502 and heavily modified.
class Uninterrupt(object):
    """
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """
    def __init__(self, sigs=(signal.SIGINT,), verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("\nInterruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None
