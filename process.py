import numpy as np
import cv2
import depthai
import time

device = depthai.Device('',False)
p = device.create_pipeline(config={
    'streams':['previewout','metaout'],
    'ai': {
        "blob_file": "depthai/resources/nn/face-detection-retail-0004/face-detection-retail-0004.blob.sh14cmx14NCE1",
        "blob_file_config": "depthai/resources/nn/face-detection-retail-0004/face-detection-retail-0004.json",
        'calc_dist_to_bb': True,
        'shaves' : 14,
        'cmx_slices' : 14,
        'NN_engines' : 1,
    },
})

if p is None:
    raise RuntimeError("Error initializing pipeline")

detections = []

class VirtualWindow:
    def __init__(self,src,width,height,camera_on=False,annotate=False):
        self.src = src
        self.width = width
        self.height = height
        self.window_width = 1920
        self.window_height = 1080
        self.x_nom = 0.5
        self.y_nom = 0.5
        self.z_nom = 0
        self.src_width = None
        self.src_height = None
        self.x = 0
        self.y = 0
        self.z = 0        #m
        self.x0 = None
        self.y0 = None
        self.max_depth = 2
        self.depth_deadband = 0.02
        self.tau = 0
        self.tau_d = 0
        self.t = time.time()
        self.frame_decimate = 60            # number of frames
        self.frame_count = 0
        self.deadband = 0                  # px deadband on position
        self.camera_on = camera_on
        self.annotate = annotate
        

    def stream(self):
        self.cap = cv2.VideoCapture(self.src)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            self.src_width = np.shape(frame)[1]
            self.src_height = np.shape(frame)[0]

        while True:
            self.get_oak_coord()
            self.play_video()
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()


    def get_oak_coord(self):
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
        detections = []
        for nnet_packet in nnet_packets:
            detections = list(nnet_packet.getDetectedObjects())
        
        x,y,z = None,None,None    
        for packet in data_packets:
            if packet.stream_name == 'previewout':
                data = packet.getData()
                data0 = data[0, :, :]
                data1 = data[1, :, :]
                data2 = data[2, :, :]
                frame = cv2.merge([data0, data1, data2])

                img_h = frame.shape[0]
                img_w = frame.shape[1]


                for detection in detections:
                    pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                    pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)
                    x = int((detection.x_min + detection.x_max)* img_w/2)
                    y = int((detection.y_min + detection.y_max)* img_h/2)
                    z = detection.depth_z
                    cv2.rectangle(frame,pt1,pt2,(0,0,255),2)
                    label = "face ({0},{1},{2:0.2f})".format(x,y,z)
                    cv2.putText(frame,label,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                
                if self.camera_on:
                    cv2.imshow('previewout', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        self.x = x
        self.y = y
        if z is not None:
            if abs(self.z - z) > self.depth_deadband:
                self.z = z

    def play_video(self):
        window = cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame',self.window_width,self.window_height)
        frame_start = time.time()
        ret, frame = self.cap.read()
        if ret:
            frame = self.crop(frame)
            frame_end = time.time()
            label = "fps: {0}".format(1/(frame_end - frame_start))
            if self.annotate:
                cv2.putText(frame,label,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            cv2.imshow('frame',frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def crop(self,frame):
        # TODO make dynamic based off window size of oak + depth max
        new_frame = frame 
        if self.x is not None and self.y is not None and self.z is not None:
            if (self.frame_count % self.frame_decimate) == 0:
                self.frame_count = 0
                x_nom = self.x / 300 # normalize
                y_nom = self.y /300

                if (abs(self.x_nom - x_nom) > self.deadband/300) or (abs(self.y_nom - y_nom) > self.deadband/300):
                    self.x_nom = x_nom            
                    self.y_nom = y_nom
                    self.width = int(self.z_nom * self.src_width)
                    self.height = int(self.z_nom * self.src_height)
                    x0 = int(self.x_nom * (self.src_width - self.width))     # pixel location 
                    y0 = int((1 - self.y_nom) * (self.src_height - self.height))
                    self.lowpass(x0,y0)

        if np.array([self.x0, self.y0, self.width, self.height]).all():
            new_frame = frame[self.y0:self.y0+self.height-1, self.x0:self.x0+self.width-1]
        return new_frame
    
    def lowpass(self,x,y):
        # low pass measurements to smooth motion
        dt = time.time() - self.t
        self.t = time.time()

        a = dt/(self.tau+dt)
        az = dt/(self.tau_d + dt)
        if self.x0 is not None and self.y0 is not None and self.z is not None:
            self.x0 = int(a*x + (1-a)*self.x0)
            self.y0 = int(a*y + (1-a)*self.y0)
            self.z_nom = az*min(self.z / self.max_depth,1) + (1-az)*self.z_nom
        else:
            self.x0 = x
            self.y0 = y

if __name__ == "__main__":
    vw = VirtualWindow('resources\AdobeStock_267854166_Video_4k_Preview.mov', 1280, 720,camera_on=False,annotate=False)
    vw.stream()

cv2.destroyAllWindows()
del p
del device