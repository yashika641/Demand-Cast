import { auth } from "./firebase";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "firebase/auth";

export async function signUp(email, password) {
  return createUserWithEmailAndPassword(auth, email, password);
}
export async function signIn(email, password) {
  return signInWithEmailAndPassword(auth, email, password);
}
export async function logout() {
  return signOut(auth);
}
