# Uility function of animatin play
# Note: Not supported by Chrome

from IPython.display import HTML

def play_animation(filename='animation.mp4'):
    """
    Play the animation file (.mp4) on iPython notebook via HTML5. 
    """
    video = open(filename, "rb").read()
    video_encoded = video.encode("base64")
    video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'\
                .format(video_encoded)
    HTML(video_tag)
