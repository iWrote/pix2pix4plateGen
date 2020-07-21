Plate_Generation. Ignore other folders.


- pix2pix and cycleGAN are both cGANS which can generate a labelled data from dataset.
- From pix2pix [paper](https://arxiv.org/pdf/1611.07004.pdf) section 6.2: **large datasets are required to colorize from black&white outlines.**
- pix2pix learns quickly from small dataset if input domain is also colorized, like [here.](https://blog.paperspace.com/content/images/2018/08/test-html-1.png)
- I'm using an [opensource dataset](https://medusa.fit.vutbr.cz/traffic/research-topics/general-traffic-analysis/holistic-recognition-of-low-quality-license-plates-by-cnn-using-track-annotated-data-iwt4s-avss-2017/) has only 300 images.
- Outlines to Image didn't work.
- So now i'm trying to colorize input. DIDN'T WORK. SIMILAR RESULTS. 
- Trying Grayscale to Grayscale conversion now. **ACCEPTABLE RESULTS.**
- Built a program to use trained model to generate arbitrary numbers. 

- CDAC dataset is low-res. Modified pix2pix architecture to work with it (reduced encoding-decoding steps). Prepared dataset.
- Will train on [opensource dataset](https://medusa.fit.vutbr.cz/traffic/research-topics/general-traffic-analysis/holistic-recognition-of-low-quality-license-plates-by-cnn-using-track-annotated-data-iwt4s-avss-2017/) and then on CDAC dataset. Then see if results are useful.

- Wrote a low-res generation script. Okay results :/
- Created a script that can be run from command line. Added a requirements.txt
- Added a flask app. Could not host it on Heroku because the pytorch dependency is too huge.

