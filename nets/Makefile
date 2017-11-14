caffe.proto:
	curl -LO "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto"

%_pb2.py: %.proto
	protoc --python_out=. $@

%.json: %.prototxt caffe_pb2.py
	python3 extract_caffe.py < $< > $@
