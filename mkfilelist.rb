
KAGGLE_DATA_DIR = "#{ENV['HOME']}/data/cifar-10-kaggle/"
TRAIN_DIR = File.join(KAGGLE_DATA_DIR, "train")
TEST_DIR = File.join(KAGGLE_DATA_DIR, "test")

labels = File.read(File.join(KAGGLE_DATA_DIR, "trainLabels.csv")).split("\n")
labels.shift

id_lut = {}
label_lut = {}
label_id = 0

labels.each do |line|
  id, label = line.split(",")
  if !label_lut.key?(label)
    label_lut[label] = label_id
    label_id += 1
  end
  id_lut[id.to_i] = label_lut[label]
end

files = Dir.entries(TRAIN_DIR).map{|file|
  if file =~ /(\d+).png/
    id = $1.to_i
    label = id_lut[id]
    if !label
      warn "#{id} label not found!!"
      exit -1
    end
    [id, File.join(TRAIN_DIR, file), label]
  else
    nil
  end
}.reject{|v| v.nil?}.sort{|a,b| a[0] <=> b[0]}
File.open("data/train.txt", "w") do |f|
  files.each do |a|
    f.puts "#{a[2]} #{a[1]}"
  end
end

files = Dir.entries(TEST_DIR).map{|file|
  if file =~ /(\d+).png/
    id = $1.to_i
    [id, File.join(TEST_DIR, file)]
  else
    nil
  end
}.reject{|v| v.nil?}.sort{|a,b| a[0] <=> b[0]}

File.open("data/test.txt", "w") do |f|
  files.each do |a|
    f.puts a[1]
  end
end
File.open("data/label_lut", "w") do |f|
  label_lut.each do |k, v|
    f.puts "#{k} #{v}"
  end
end
