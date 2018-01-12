var timeoutPeriod = 50;
var imageURI = './sources/cam.png';

setInterval(function(){document.getElementById("liveImg").setAttribute("src", imageURI + '?d=' + Date.now())}, timeoutPeriod);
