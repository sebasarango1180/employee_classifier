var timeoutPeriod = 50;
var imageURI = './sources/cam.png';
var x=0, y=0;
var img = new Image();
img.onload = function() {
    var canvas = document.getElementById("liveImg");
    var context = canvas.getContext("2d");

    context.drawImage(img, x, y);
    x+=20; y+=20;
    setTimeout(timedRefresh,timeoutPeriod);
};

function timedRefresh() {
    // just change src attribute, will always trigger the onload callback
    img.src = imageURI + '?d=' + Date.now();
}