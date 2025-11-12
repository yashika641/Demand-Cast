// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";
import { getDatabase } from "firebase/database";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC3LyVjlRjGNYAomlPjCFFTpR4BkUdkG6w",
  authDomain: "demandcast.firebaseapp.com",
  projectId: "demandcast",
  storageBucket: "demandcast.firebasestorage.app",
  messagingSenderId: "332534777021",
  appId: "1:332534777021:web:526d9b5966ef2003d123d4",
  measurementId: "G-7NMXFQ4B5P"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const database =getDatabase(app);
export const auth = getAuth(app);

export async function signIn(email, password) {
  const userCredential = await signInWithEmailAndPassword(auth, email, password);
  return userCredential.user;
}