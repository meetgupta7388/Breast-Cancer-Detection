import { initializeApp } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-app.js";
import { getAuth, 
         GoogleAuthProvider,
         connectAuthEmulator } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-firestore.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyD7C8yMc6fifnZ-Q3MyvFjAI9TuInJW3bQ",
  authDomain: "oncodetect-7d355.firebaseapp.com",
  projectId: "oncodetect-7d355",
  storageBucket: "oncodetect-7d355.firebasestorage.app",
  messagingSenderId: "870630394900",
  appId: "1:870630394900:web:009a8f3154d6a675522b01",
  measurementId: "G-M2MV9EM243"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Configure Google Auth Provider
provider.setCustomParameters({
    prompt: 'select_account'
});

const db = getFirestore(app);

export { auth, provider, db };