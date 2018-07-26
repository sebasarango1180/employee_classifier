var timeoutPeriod = 100;
var imageURI = './sources/cam.jpg';

setInterval(function(){document.getElementById("liveImg").setAttribute("src", imageURI + '?d=' + Date.now())}, timeoutPeriod);
