import {useEffect} from "react";

function AuthAdmin() {
    return (
        <>
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
                    Форма авторизации в панель администратора parfumtim.ru
                </h1>
                <form method="POST" action="https://parfumtim.ru/Source/AdminMethods/AuthAdmin.php" style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    padding: '30px',
                    background: '#fff',
                    borderRadius: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                }}>
                    <input
                        type="password"
                        name="password"
                        placeholder="Введите пароль"
                        required
                        style={{
                            padding: '20px',
                            width: '220px',
                            marginBottom: '15px',
                            border: '1px solid #ccc',
                            borderRadius: '6px',
                            fontSize: '16px'
                        }}
                    />
                    <input
                        className="button"
                        type="submit"
                        value="Войти"
                        style={{
                            backgroundColor: '#007bff',
                            color: 'white',
                            border: 'none',
                            padding: '10px 20px',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '16px'
                        }}
                    />
                </form>
            </div>
        </>
    )
}

export default AuthAdmin;