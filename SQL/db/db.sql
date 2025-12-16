CREATE DATABASE riskai;

CREATE TABLE datasets (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path VARCHAR(512) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_dataset_name (name),
    UNIQUE KEY uq_dataset_path (path)
) ENGINE=InnoDB;


CREATE TABLE models (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path VARCHAR(512) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_model_name (name),
    UNIQUE KEY uq_model_path (path)
) ENGINE=InnoDB;

CREATE TABLE model_datasets (
    model_id BIGINT UNSIGNED NOT NULL,
    dataset_id BIGINT UNSIGNED NOT NULL,

    PRIMARY KEY (model_id, dataset_id),

    CONSTRAINT fk_md_model
        FOREIGN KEY (model_id) REFERENCES models(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_md_dataset
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        ON DELETE CASCADE
) ENGINE=InnoDB;
