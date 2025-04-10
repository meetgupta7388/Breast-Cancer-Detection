// Handle profile image upload
document.getElementById('image-upload')?.addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert('Image size should be less than 5MB');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch('/upload-profile-image', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('profile-image').src = data.image_url;
        } else {
            const data = await response.json();
            alert(data.error || 'Failed to upload image');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload image');
    }
});

// Handle profile form submission
document.getElementById('profile-form')?.addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/update-profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            alert('Profile updated successfully');
            // Update the welcome message in the dashboard
            if (data.first_name) {
                localStorage.setItem('userFirstName', data.first_name);
            }
        } else {
            const error = await response.json();
            alert(error.message || 'Failed to update profile');
        }
    } catch (error) {
        console.error('Update error:', error);
        alert('Failed to update profile');
    }
}); 