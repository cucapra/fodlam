# Network specifications from the Web.

NETS := VGG_CNN_M_deploy VGG_ILSVRC_16_layers_deploy alexnet_train_val

.PHONY: all clean
all: $(NETS:%=%.json)
clean:
	rm -f $(NETS:%=%.prototxt) $(NETS:%=%.json)

VGG_CNN_M_deploy.prototxt:
	curl -LO "https://gist.githubusercontent.com/ksimonyan/f194575702fae63b2829/raw/6516d9be2064680697ee6791d8de77cfc0dd0990/VGG_CNN_M_deploy.prototxt"

VGG_ILSVRC_16_layers_deploy.prototxt:
	curl -LO "https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt"

alexnet_train_val.prototxt:
	curl -L -o $@ "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/train_val.prototxt"


# Conversion infrastructure.

caffe.proto:
	curl -LO "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto"

%_pb2.py: %.proto
	protoc --python_out=. $@

%.json: %.prototxt caffe_pb2.py
	python3 extract_caffe.py < $< > $@
