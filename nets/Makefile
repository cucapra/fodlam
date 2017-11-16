# Network specifications from the Web.

NETS := VGG_CNN_M_deploy VGG_ILSVRC_16_layers_deploy alexnet_deploy \
	faster_rcnn_vgg16 faster_rcnn_vggm

.PHONY: all clean
all: $(NETS:%=%.json)
clean:
	rm -f $(NETS:%=%.prototxt) $(NETS:%=%.json)

# Prototxt from the web.

VGG_CNN_M_deploy.prototxt:
	curl -LO "https://gist.githubusercontent.com/ksimonyan/f194575702fae63b2829/raw/6516d9be2064680697ee6791d8de77cfc0dd0990/VGG_CNN_M_deploy.prototxt"

VGG_ILSVRC_16_layers_deploy.prototxt:
	curl -LO "https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt"

alexnet_deploy.prototxt:
	curl -L -o $@ "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt"

faster_rcnn_vgg16.prototxt:
	curl -L -o $@ "https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"

faster_rcnn_vggm.prototxt:
	curl -L -o $@ "https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"

# Conversion tool.

%.json: %.prototxt
	python2 extract.py $< > $@
