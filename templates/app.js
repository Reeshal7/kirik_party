const firebaseConfig = {
  apiKey: "AIzaSyDp8ZKMT7xNZCpTJE449VKu-K3i_kZgmaM",
  authDomain: "kirik-party-11088.firebaseapp.com",
  projectId: "kirik-party-11088",
  storageBucket: "kirik-party-11088.firebasestorage.app",
  messagingSenderId: "861588860533",
  appId: "1:861588860533:web:93441d80922f2903d6123c",
  measurementId: "G-VDD05FKR5H"
};
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();
function saveEmotion() {
  db.collection("emotions").add({
    studentName: "John Doe",
    emotion: "Happy",
    confidence: 0.95,
    timestamp: new Date()
  })
  .then(() => {
    alert("Emotion saved successfully!");
  })
  .catch((error) => {
    console.error("Error saving emotion: ", error);
  });
}