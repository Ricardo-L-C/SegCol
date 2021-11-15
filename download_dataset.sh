rsync --progress --verbose rsync://78.46.86.149:873/danbooru2020/metadata.json.tar.xz ./dataset/
mkdir ./dataset/metadata
tar -xf ./dataset/metadata.json.tar.xz -C ./dataset
rm dataset/metadata.json.tar.xz

rsync --recursive --times --verbose rsync://78.46.86.149:873/danbooru2020/512px/ ./dataset/512px/

