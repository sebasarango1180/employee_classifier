var timeoutPeriod = 100;
var imageURI = './sources/cam.png';
//var x=0, y=0;
var img = new Image();
var but = document.getElementById("sub");

function sleep(milliseconds) {
  var start = new Date().getTime();
  for (var i = 0; i < 1e7; i++) {
    if ((new Date().getTime() - start) > milliseconds){
      break;
    }
  }
}

function show_frames(){

    //img = new Image();

    var canvas = document.getElementById("liveImg");

    var context = canvas.getContext("2d");

    context.drawImage(img, 0, 0);
                           //x+=20; y+=20;
    setTimeout(timedRefresh,timeoutPeriod);
};

img.onload = function() {

    show_frames();

};

function show_cam(){

    sleep(1000);

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
