var message;

var compiled = []
var totalInput;

function reset(){
  nilaiMataKuliah=[];
  compiled=[];
  var list = document.getElementsByClassName("modal-body")[0];
  for(var i=0;i<7;i++){
    list.getElementsByClassName("nilai")[i].value =0;
    list.getElementsByClassName("nilai")[i].disabled=false;
    if(i<5){
      list.getElementsByClassName("judul")[i].innerHTML=" ";
      list.getElementsByClassName("abstrak")[i].innerHTML=" ";
    }
  }
}
function getData() {
  var inputElements = document.getElementsByClassName('mk');
  for(var i=0; inputElements[i]; i++){
    if(inputElements[i].checked){
      compiled.push(inputElements[i].value);
    }
  }
  var list = document.getElementsByClassName("modal-body")[0];
  //mendapatkan seluruh mata kuliah yang sudah dipilih
  for(var i=0;i<compiled.length;i++){
    list.getElementsByClassName("matkul")[i].innerHTML = compiled[i];
    totalInput=i+1
  }
  //mengganti paragraph dari matkul ke i menjadi - dan mendisable text area
  for(var i = totalInput;i<7;i++){
    list.getElementsByClassName("matkul")[i].innerHTML = "-"
    list.getElementsByClassName("nilai")[i].disabled=true;
  }
}

function getNilai(){
  var nilai = []
  var inputElements = document.getElementsByClassName("nilai");

  for(var i=0;i<totalInput;i++){
    //melakukan validasi ketika inputan nilai 0 agar tidak error saat perhitungan
    // list.getElementsByClassName("judul")[x].innerHTML=data_output[i];
    if(inputElements[i].value=='0'){
      compiled.push(0.01)
      console.log(compiled) //buat ngeprint
    }
    else{
      compiled.push(parseFloat(inputElements[i].value));
      console.log(compiled)
    }
  }

  for (var i = 0; i < compiled.length; i++) {
    
  }
  //mengirim data ke back-end untuk diolah
  calltoServer(compiled);
}

function calltoServer(Value){
  //variabel untuk merequest data ke back-end
  var xhttp = new XMLHttpRequest();
  console.log(Value.length);

  var data_count = Value.length/2;
  var list = document.getElementsByClassName("modal-body")[1];
  var x=0;

  for (var i = 0; i < data_count; i++) {
    list.getElementsByClassName("judul")[i].innerHTML=Value[i];
  }

  for (var i = data_count; i < Value.length; i++) {
    list.getElementsByClassName("abstrak")[x].innerHTML=Value[i];
    x++;
  }

  //fungsi ketika berhasil mendapat response dari backend
  xhttp.onreadystatechange = function() {
    if (this.readyState == XMLHttpRequest.DONE) {
      message=xhttp.responseText
      data_output=JSON.parse(message)
      console.log(data_output)
      var list = document.getElementsByClassName("modal-body")[1];
      var x=0;
      for(var i = 0;i<10;i++){
        if(i%2==0){
          list.getElementsByClassName("judul")[x].innerHTML=data_output[i];
          // untuk menulis di list tabel
        }
        else{
          list.getElementsByClassName("abstrak")[x].innerHTML=data_output[i];
          x++;
        }
      }
    }
  };
  //mulai membuka koneksi ke back-end dengan method post ke link url
  xhttp.open("POST", "http://127.0.0.1:5000/byacademicprofile", true);
  xhttp.setRequestHeader( "Content-Type", "application/json" );
  xhttp.send(JSON.stringify(Value));
}

function checkcontrol(j){
  var totalchecked=0;
  for(var i=0;i<document.formMataKuliah.rGroup.length;i++){
    if(document.formMataKuliah.rGroup[i].checked){
      totalchecked=totalchecked+1;
    }
    if(totalchecked>7){
      alert("You can only choose 7 courses")
      document.formMataKuliah.rGroup[j].checked=false;
      return false;
    }
  }
}

