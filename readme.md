Original repo had data interns cannot share online. Redacted. Cue spy music.

#### Problem: Creating fake (but realistic) images to train an automated number plate recognition program. Because we don't have enough real ones. Procedural generation is not allowed. 

Step 0: Panick.   
Step 1: Read about all this AI jazz.   
Step 2: Get intimidated by tensorflow and do pyTorch tutorials.  
Step 3: Follow a tutorial on classifying images Cats v/s Dogs.  
Step 4: Minimalistic GAN implementaion tutorial.  
Step 5: DIY mini-project: Generate images like MNIST handwritten digits with a GAN.  
Step 6: Research (a.k.a. intensively google) GAN architectures smarter people have come up with.   
Step 7: Carefully read the paper on pix2pix.   
Step 8: Follow an excellent tensorflow tutorial, translating to pyTorch as you go. Regret Step 2.   
Step 9: Try a bunch on things. Tweak. Rinse. Repeat. (Longest step. See .pptx)   
Step 10: Done.   

#### Solution (see .pptx):
- Use photographs of [car butts](https://medusa.fit.vutbr.cz/traffic/research-topics/general-traffic-analysis/holistic-recognition-of-low-quality-license-plates-by-cnn-using-track-annotated-data-iwt4s-avss-2017/)   
- Char to monochrome.  
- Wipe said butts. Brand with a new number.  
- Reshade/Rescorch.  

![](https://github.com/iWrote/pix2pix4plateGen/blob/master/PICTURES%20Yay/final%20solution.PNG)
![](https://github.com/iWrote/pix2pix4plateGen/blob/master/PICTURES%20Yay/SUCCESS__pug_plate.png)  
![](https://github.com/iWrote/pix2pix4plateGen/blob/master/PICTURES%20Yay/LOW-RES-PLATE-GEN.gif)  
Threw in a cropped-image generation script on request. 

#### Other kewl stuff:
See .pptx and notebooks for implementation details.   
Step 5 result!  
![](https://github.com/iWrote/pix2pix4plateGen/blob/master/phase%201%20practice%20notebooks/mnistGanSuccess.gif)
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  
.  

        
         
         
          
            
            







OLD LOGS
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

