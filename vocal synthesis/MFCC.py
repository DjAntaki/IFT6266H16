from fuel.transformers import Flatten
from fuel.transformers.sequences import Window
from fuel.transformers import Mapping, Batch, FilterSources, Merge, ForceFloatX
import features


def mfcc(array,samplerate=16000, winlen=0.025,winstep=0.01,numcep=13, nfilt=26,nfft=256, appendEnergy=True):
    array = array[0] # Because of FilterSources returns a list
    feats = features.mfcc(array, samplerate, winlen, winstep, numcep, nfilt, nfft, appendEnergy = appendEnergy)
    return [feats]

def get_stream(source_window=4000, target_window=1000):
    from fuel.datasets.youtube_audio import YouTubeAudio
    data = YouTubeAudio('XqaJ2Ol5cC4')
    train_stream = data.get_example_stream()
    train_stream = ForceFloatX(train_stream)
    window_stream = Window(0,source_window, target_window, overlapping=False, data_stream=train_stream)
    source_stream = FilterSources(window_stream, sources=('features',))
    feats_stream = Mapping(source_stream, mfcc)
    targets_stream = FilterSources(window_stream, sources=('targets',))
    targets_stream = Flatten(targets_stream)
    stream = Merge((feats_stream,targets_stream),sources = ('features','targets'))
    return stream

#if __name__ == '__main__':
#    from fuel.datasets.youtube_audio import YouTubeAudio
#    data = YouTubeAudio('XqaJ2Ol5cC4')
#    train_stream = data.get_example_stream()
#    train_stream = ForceFloatX(train_stream)
#    window_stream = Window(0,source_window, target_window, overlapping=False, data_stream=train_stream)
#    source_stream = FilterSources(window_stream, sources=('features',))
#    feats_stream = Mapping(source_stream, mfcc)
#    targets_stream = FilterSources(window_stream, sources=('targets',))
#    train_stream = Merge((feats_stream,targets_stream),sources = ('features','targets'))
