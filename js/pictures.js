var timeoutPeriod = 50;
var imageURI = './sources/cam.png';
var x=0, y=0;
var img = new Image();

function show_frames(){
    var canvas = document.getElementById("liveImg");
    var context = canvas.getContext("2d");

    context.drawImage(img, 0, 0);
                           //x+=20; y+=20;
    setTimeout(timedRefresh,timeoutPeriod);
     };


function timedRefresh() {
    // just change src attribute, will always trigger the onload callback
    img.src = imageURI + '?d=' + Date.now();
}

img.onload = function() {

show_frames();

};