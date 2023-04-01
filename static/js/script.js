var message;
var compiled = []
var totalInput;

function reset() {
  nilaiMataKuliah = [];
  compiled = [];
  var list = document.getElementsByClassName("modal-body")[0];
  for (var i = 0; i < 7; i++) {
    list.getElementsByClassName("nilai")[i].value = 0;
    list.getElementsByClassName("nilai")[i].disabled = false;
    if (i < 5) {
      list.getElementsByClassName("judul")[i].innerHTML = " ";
      list.getElementsByClassName("abstrak")[i].innerHTML = " ";
    }
  }
}
function getData() {
  var inputElements = document.getElementsByClassName('mk');
  for (var i = 0; inputElements[i]; i++) {
    if (inputElements[i].checked) {
      compiled.push(inputElements[i].value);
    }
  }

  console.log(compiled)

  var list = document.getElementsByClassName("modal-body")[0];
  //mendapatkan seluruh mata kuliah yang sudah dipilih
  for (var i = 0; i < compiled.length; i++) {
    list.getElementsByClassName("matkul")[i].innerHTML = compiled[i];
    totalInput = i + 1
  }
  //mengganti paragraph dari matkul ke i menjadi - dan mendisable text area
  for (var i = totalInput; i < 7; i++) {
    list.getElementsByClassName("matkul")[i].innerHTML = "-"
    list.getElementsByClassName("nilai")[i].disabled = true;
  }
}

function getNilai() {
  var inputElements = document.getElementsByClassName("nilai");

  for (var i = 0; inputElements[i]; i++) {
    if (inputElements[i].value && inputElements[i].value != "none") {
      compiled[i] = {
        "matkul": compiled[i],
        "nilai": inputElements[i].value
      }
    }
  }
  console.log(compiled)

  //mengirim data ke back-end untuk diolah
  calltoServer(compiled);
  compiled = []
}

function calltoServer(value) {
  //variabel untuk merequest data ke back-end
  var xhttp = new XMLHttpRequest();

  //mulai membuka koneksi ke back-end dengan method post ke link url
  xhttp.open("POST", "/result_byacademicprofile", true);
  xhttp.setRequestHeader("Content-Type", "application/json");
  xhttp.send(JSON.stringify(value));
  compiled = []
}

function checkcontrol(j) {
  var totalchecked = 0;
  for (var i = 0; i < document.formMataKuliah.rGroup.length; i++) {
    if (document.formMataKuliah.rGroup[i].checked) {
      totalchecked = totalchecked + 1;
    }
    if (totalchecked > 7) {
      alert("You can only choose 7 courses")
      document.formMataKuliah.rGroup[j].checked = false;
      return false;
    }
  }
}
