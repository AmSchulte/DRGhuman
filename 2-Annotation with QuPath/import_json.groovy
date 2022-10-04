import qupath.lib.objects.PathObjects
def gson = GsonTools.getInstance(true)

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
def import_dir = "E:/python/hubmap/annotation/kaggle_hubmap/data/test/ext/iafos"
def fname = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).substring(0,9)
def name = String.format("%s/%s.json", import_dir, fname)

println fname

def json = new File(name).text
//def json = new File("/Users/matjes/hubmap-kidney-segmentation/train/0486052bb.json").text



// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType();
//def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType();
def deserializedAnnotations = gson.fromJson(json, type)

// Set the annotations to have a different name (so we can identify them) & add to the current image
// deserializedAnnotations.eachWithIndex {annotation, i -> annotation.setName('New annotation ' + (i+1))}   # --- THIS WON"T WORK IN CURRENT VERSION
addObjects(deserializedAnnotations)   
                                                                                                                                              