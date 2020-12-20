Usage:
  Sobel Test [OPTION...]

  -s, --show [=arg(=0)]     Realtime video visualization mode, ex:
                            --show=video.mp4

  -p, --photo arg           Proccess a photo file, ex: photo.jpg

  -t, --time arg            Shows time dif between cpu and cuda time for a
                            given video file, ex: testvideo.mp4

  -o, --out [=arg(=o.jpg)]  Output name for photo file, ex: out.jpg

  -h, --help                Print usage

Examples:

cudaConvolution -v testData/testvideo.mp4 -g

cudaConvolution -p testData/mountain.jpg -g

cudaConvolution -s

cudaConvolution -show=testData/testvideo.mp4



Note:

When in --show mode, you can hit the ESC key to close the app.


APIs used:

 -cxxopts API by: https://github.com/jarro2783/cxxopts
