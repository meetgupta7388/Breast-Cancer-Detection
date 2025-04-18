// Initialize Firebase Auth
const auth = firebase.auth();

// Handle regular email/password login
document.getElementById('sign-in-btn')?.addEventListener('click', async function() {
    const email = document.getElementById('email-input').value;
    const password = document.getElementById('password-input').value;
    const emailError = document.getElementById('email-error-message');
    const passwordError = document.getElementById('password-error-message');

    try {
        const userCredential = await auth.signInWithEmailAndPassword(email, password);
        const idToken = await userCredential.user.getIdToken();
        
        // Send token to backend
        const response = await fetch('/auth', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${idToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            window.location.href = data.redirect;
        } else {
            const data = await response.json();
            emailError.textContent = data.error || 'Invalid email or password';
        }
    } catch (error) {
        console.error('Login error:', error);
        emailError.textContent = error.message;
    }
});

// Handle Google Sign In/Sign Up
const handleGoogleAuth = async (event) => {
    const provider = new firebase.auth.GoogleAuthProvider();
    const errorElement = document.getElementById('google-signin-error-message');
    
    // Force account selection
    provider.setCustomParameters({
        prompt: 'select_account'
    });
    
    try {
        const result = await auth.signInWithPopup(provider);
        const idToken = await result.user.getIdToken();
        
        // Send token to backend
        const response = await fetch('/auth', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${idToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            window.location.href = data.redirect;
        } else {
            const data = await response.json();
            errorElement.textContent = data.error || 'Authentication failed';
        }
    } catch (error) {
        console.error('Google auth error:', error);
        errorElement.textContent = error.message;
    }
};

// Add event listeners for Google buttons
document.getElementById('sign-in-with-google-btn')?.addEventListener('click', handleGoogleAuth);
document.getElementById('sign-up-with-google-btn')?.addEventListener('click', handleGoogleAuth);

// Handle Create Account
document.getElementById('create-account-btn')?.addEventListener('click', async function() {
    const email = document.getElementById('email-input').value;
    const password = document.getElementById('password-input').value;
    const emailError = document.getElementById('email-error-message');
    const passwordError = document.getElementById('password-error-message');

    try {
        const userCredential = await auth.createUserWithEmailAndPassword(email, password);
        const idToken = await userCredential.user.getIdToken();
        
        // Send token to backend
        const response = await fetch('/auth', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${idToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            window.location.href = data.redirect;
        } else {
            const data = await response.json();
            emailError.textContent = data.error || 'Failed to create account';
        }
    } catch (error) {
        console.error('Signup error:', error);
        emailError.textContent = error.message;
    }
});

// Handle Forgot Password
document.getElementById('reset-password-form')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    const email = document.getElementById('email-input').value;
    const emailError = document.getElementById('email-error-message');

    try {
        await auth.sendPasswordResetEmail(email);
        alert('Password reset email sent. Please check your inbox.');
        window.location.href = '/login';
    } catch (error) {
        console.error('Reset password error:', error);
        emailError.textContent = error.message;
    }
});
