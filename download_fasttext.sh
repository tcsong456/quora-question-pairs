mkdir -p artifacts
cd artifacts
echo 'Downloading crawl-300-2M.bin fasttext embedding'
wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
echo 'Unzipping'
gunzip *.gz 2>/dev/null
