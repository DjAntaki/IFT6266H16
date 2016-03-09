#wrapping python-speech-feature librairy with SourwiseTransformer and ExpectsAxisLabels from fuel.transformers



import numpy

from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
import features

class MFCC(SourcewiseTransformer, ExpectsAxisLabels):
    """  """
    def __init__(self, data_stream, winlen=10,winstep=5,numcep=13,
          nfilt=26,nfft=256, appendEnergy=True, **kwargs):
        self.winlen = winlen
        self.winstep = winstep
        self.nfilt=nfilt
        self.nfft=nfft
        self.numcep=numcep
        
#        self.lowfreq=0
#        self.highfreq=None
#        self.preemph=0.97,
#        self.ceplifter=22
#        self.appendEnergy=True
        
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(MFCC, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'time', 'feature'),
                                self.data_stream.axis_labels[source_name],
                                source_name)

        print(source.shape)
        print('aaaa')

        if isinstance(source, numpy.ndarray) and source.ndim == 3:

            raise Exception
        
        elif all(isinstance(b, numpy.ndarray) and b.ndim == 2 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('time', 'feature'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        
        print(example.shape)


        
        if not isinstance(example, numpy.ndarray) or example.ndim != 2:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 2")
                


        feats = features.mfcc(example, winlen = self.winlen,
        winstep = self.winstep, nfilt=self.nfilt, nfft=self.nfft)

        return feats


