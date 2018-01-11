var timeoutPeriod = 100;
var imageURI = './sources/cam.png';
//var x=0, y=0;
var img = new Image();
var but = document.getElementById("sub");


setInterval(function(){document.getElementById("liveImg").setAttribute("src", imageURI + '?d=' + Date.now())}, timeoutPeriod);
