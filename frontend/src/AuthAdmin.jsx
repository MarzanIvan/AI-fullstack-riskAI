import { useState } from "react";

function AuthAdmin() {
    const [password, setPassword] = useState("");
    const [message, setMessage] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const formData = new URLSearchParams();
            formData.append("password", password);

            const response = await fetch("/php_api/AuthAdmin.php", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                body: formData.toString()
            });

            const result = await response.json();

            if (result.success) {
                // Успешная авторизация — редирект на AdminPanel.php
                window.location.href = "/php_api/AdminPanel.php";
            } else {
                setMessage("Неверный пароль");
            }
        } catch (err) {
            console.error(err);
            setMessage("Ошибка соединения с сервером");
        }
    };

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                height: "100vh",
                fontFamily: "Arial, sans-serif"
            }}
        >
            <h1
                style={{
                    marginBottom: "20px",
                    fontSize: "24px",
                    color: "#333",
                    textAlign: "center"
                }}
            >
                Форма авторизации в панель администратора riskai.ru___
            </h1>

            <form
                onSubmit={handleSubmit}
                style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    padding: "30px",
                    background: "#fff",
                    borderRadius: "12px",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.15)"
                }}
            >
                <input
                    type="password"
                    placeholder="Введите пароль"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    style={{
                        padding: "20px",
                        width: "220px",
                        marginBottom: "15px",
                        border: "1px solid #ccc",
                        borderRadius: "6px",
                        fontSize: "16px"
                    }}
                />

                <input
                    className="button"
                    type="submit"
                    value="Войти"
                    style={{
                        backgroundColor: "#007bff",
                        color: "white",
                        border: "none",
                        padding: "10px 20px",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontSize: "16px"
                    }}
                />
            </form>

            {message && (
                <p style={{ marginTop: "15px", color: "red", fontSize: "16px" }}>
                    {message}
                </p>
            )}
        </div>
    );
}

export default AuthAdmin;
