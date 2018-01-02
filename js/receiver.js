var socket = io();
    socket.on('stream',function(image){
      var img = document.getElementById("liveImg");
      img.src = image;
    });