import cv2
import depthai
import sys
sys.path.append('depthai')
import depthai_demo
import consts.resource_paths

sys.path.append('depthai/depthai_helpers')
import mobilenet_ssd_handler

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

nn2depth = True 
nn_depth = device.get_nn_to_depth_bbox_mapping()

def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

detections = []


if __name__ == "__main__":
    while True:
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

        for nnet_packet in nnet_packets:
            detections = list(nnet_packet.getDetectedObjects())
        
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
                    cv2.rectangle(frame,pt1,pt2,(0,0,255),2)
                    label = "face ({0},{1},{2:0.2f})".format(x,y,detection.depth_z)
                    cv2.putText(frame,label,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                    
                cv2.imshow('previewout', frame)

        if cv2.waitKey(1) == ord('q'):
            break
del p
del device