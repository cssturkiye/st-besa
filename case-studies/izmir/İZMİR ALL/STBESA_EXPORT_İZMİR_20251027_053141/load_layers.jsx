var folder = new Folder('<your-dir>\ST-BESA\case-studies\izmir\İZMİR ALL\STBESA_EXPORT_İZMİR_20251027_053141');
var files = folder.getFiles(/\.png$/i).sort(function (a, b) { return (decodeURI(a.name) > decodeURI(b.name)) ? 1 : -1; });
if (files.length > 0) {
  var base = app.open(files[0]);
  base.activeLayer.name = decodeURI(files[0].name.replace(/\.png$/i, ''));
  for (var i = 1; i < files.length; i++) {
    var im = app.open(files[i]); im.selection.selectAll(); im.selection.copy(); im.close(SaveOptions.DONOTSAVECHANGES); base.paste(); base.activeLayer.name = decodeURI(files[i].name.replace(/\.png$/i, ''));
  }
  base.resizeImage(undefined, undefined, 600, ResampleMethod.NONE);
}
