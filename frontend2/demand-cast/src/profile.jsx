import { auth } from "./firebase";

const user = auth.currentUser;
if (user) {
  const idToken = await user.getIdToken();

  await fetch("http://localhost:8000/profile", {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${idToken}`,
    },
  });
}
