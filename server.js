const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const nodemailer = require('nodemailer');
const axios = require('axios');
const dotenv = require('dotenv');
const { execSync} = require('child_process');
const { spawnSync } = require('child_process');

let Mistral;
try {
  const MistralModule = require('@mistralai/mistralai');
  Mistral = MistralModule.Mistral;
} catch (e) {
  console.log('⚠ @mistralai/mistralai not installed - Mistral features disabled');
  Mistral = null;
}

const backendEnvPath = path.resolve(__dirname, '.env');
const starGenAIEnvPath = path.resolve(__dirname, '../backend/StarGenAI/.env');



dotenv.config({ path: backendEnvPath });
dotenv.config({ path: starGenAIEnvPath, override: false });
console.log('PYTHON_EXE =', process.env.PYTHON_EXE);
console.log('MISTRAL_API_KEY présente =', !!process.env.MISTRAL_API_KEY);
const app = express();
let PORT = process.env.PORT || 5000;
const FASTAPI_HOST = process.env.FASTAPI_HOST || '127.0.0.1';
const FASTAPI_PORT = process.env.FASTAPI_PORT || '8001';
const FASTAPI_URL = `http://${FASTAPI_HOST}:${FASTAPI_PORT}/api/chat`;


function formatClientProfile(client) {
  return `
- Nom : ${client.Nom_Client || client.Client_Name}
- Âge : ${client.Age_Client || client.Client_Age}
- Genre : ${client.Genre_Client || client.Client_Gender}
- Localisation : ${client.Ville_Client || client.Client_Location}
- Type de police : ${client.Type_Contrat || client.Policy_Type}
- Début police : ${client.Date_Debut_Contrat || client.Policy_Start_Date}
- Fin police : ${client.Date_Fin_Contrat || client.Policy_End_Date}
- Prime annuelle : ${client.Prime_TND || client.Policy_Premium_USD} TND
- Statut sinistre : ${client.Statut_Sinistre || client.Claim_Status}
- Montant sinistre : ${client.Montant_Sinistre_TND || client.Claim_Amount_USD} TND
- Dernière interaction : ${client.Derniere_Interaction || client.Last_Interaction}
- Score de risque : ${client.Score_Risque || client.Risk_Score}
- Probabilité renouvellement : ${client.Probabilite_Renouvellement || client.Renewal_Probability}
- Indicateur de fraude : ${client.Indicateur_Fraude || client.Fraud_Risk_Flag}
- Score Cross-Sell : ${client.Score_CrossSell || client.Cross_Sell_Score}
- Source : ${client.Source_Prospect || client.Lead_Source}
- Feedback : ${client.Retour_Client || client.Client_Feedback}
`;
}

const fs = require('fs');
const multer = require('multer');

// Make uploads directory if not exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const safeName = file.originalname.replace(/[^a-zA-Z0-9.\-\_]/g, '_');
    cb(null, `${uniqueSuffix}-${safeName}`);
  }
});
const upload = multer({ storage });



function normalizeValue(value) {
  if (value === undefined || value === null) return '';
  return value.toString().trim().replace(/^\uFEFF/, '');
}

function parseCsvLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (inQuotes) {
      if (char === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        current += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ',') {
        values.push(current);
        current = '';
      } else {
        current += char;
      }
    }
  }

  values.push(current);
  return values;
}

function parseCsv(content) {
  const lines = content.split(/\r?\n/).filter((line) => line.trim() !== '');
  if (!lines.length) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const fields = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = normalizeValue(fields[index]);
    });
    return row;
  });
}

function titleCase(string) {
  const clean = normalizeValue(string).toLowerCase();
  return clean ? clean.charAt(0).toUpperCase() + clean.slice(1) : '';
}

function mapFraudRisk(value) {
  const normalized = normalizeValue(value).toLowerCase();
  if (normalized === 'faible') return 'Low';
  if (normalized === 'moyen') return 'Medium';
  if (normalized === 'élevé' || normalized === 'eleve') return 'High';
  return titleCase(value);
}

function mapLeadSource(value) {
  const normalized = normalizeValue(value).toLowerCase();
  if (normalized === 'agent') return 'Agent';
  if (normalized === 'site web') return 'Web';
  if (normalized === 'réseaux sociaux' || normalized === 'reseaux sociaux') return 'Web';
  if (normalized === 'recommandation') return 'Referral';
  if (normalized === 'broker' || normalized === 'courtier') return 'Broker';
  return titleCase(value);
}

function safeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function buildStarGenAIData(rows) {
  const withCleanFlag = rows.map((row) => ({
    ...row,
    Client_Name: row.Client_Name || row.Nom_Client,
    Policy_Type: row.Policy_Type || row.Type_Contrat,
    Fraud_Risk_Flag: row.Indicateur_Fraude ? mapFraudRisk(row.Indicateur_Fraude) : mapFraudRisk(row.Fraud_Risk_Flag),
    Claim_Status: titleCase(row.Statut_Sinistre || row.Claim_Status),
    Lead_Source: mapLeadSource(row.Source_Prospect || row.Lead_Source),
    Cross_Sell_Score: safeNumber(row.Score_CrossSell || row.Cross_Sell_Score),
    Renewal_Probability: safeNumber(row.Probabilite_Renouvellement || row.Renewal_Probability),
    Risk_Score: safeNumber(row.Score_Risque || row.Risk_Score),
    Policy_Premium_USD: safeNumber(row.Prime_TND || row.Policy_Premium_USD),
    Claim_Amount_USD: safeNumber(row.Montant_Sinistre_TND || row.Claim_Amount_USD),
  }));

  const policyTypes = [...new Set(withCleanFlag.map((row) => row.Policy_Type).filter(Boolean))];
  const leadSources = ['Agent', 'Web', 'Broker', 'Referral'];

  const riskByPolicy = policyTypes.map((policy) => {
    const group = withCleanFlag.filter((row) => row.Policy_Type === policy);
    const totalPremium = group.reduce((sum, row) => sum + row.Policy_Premium_USD, 0);
    const totalClaim = group.reduce((sum, row) => sum + row.Claim_Amount_USD, 0);
    const averageRenewal = group.reduce((sum, row) => sum + row.Renewal_Probability, 0) / Math.max(group.length, 1);
    const ratio = totalPremium > 0 ? totalClaim / totalPremium : 0;
    let status = 'Optimisé';
    if (ratio > 0.4) status = 'Alerte';
    else if (ratio > 0.25) status = 'Stable';

    return {
      policy,
      exposure: Math.round(totalPremium),
      sp_ratio: `${Math.round(ratio * 100)}%`,
      growth: `${Math.round((averageRenewal - 0.5) * 1000) / 10}%`,
      status,
    };
  });

  const leadSourceComparison = leadSources.map((source) => {
    const group = withCleanFlag.filter((row) => row.Lead_Source === source);
    return {
      source,
      totalPremium: Math.round(group.reduce((sum, row) => sum + row.Policy_Premium_USD, 0)),
      totalClaims: Math.round(group.reduce((sum, row) => sum + row.Claim_Amount_USD, 0)),
    };
  });

  const fraudDistribution = withCleanFlag.reduce(
    (acc, row) => {
      const key = row.Fraud_Risk_Flag;
      if (key === 'High' || key === 'Medium' || key === 'Low') {
        acc[key] += 1;
      }
      return acc;
    },
    { Low: 0, Medium: 0, High: 0 }
  );

  const renewalHeatmap = policyTypes.map((policy) => {
    const rowData = { policy, Agent: 0, Web: 0, Broker: 0, Referral: 0 };
    const group = withCleanFlag.filter((item) => item.Policy_Type === policy);
    leadSources.forEach((source) => {
      const sourceGroup = group.filter((item) => item.Lead_Source === source);
      rowData[source] =
        sourceGroup.length > 0
          ? sourceGroup.reduce((sum, item) => sum + item.Renewal_Probability, 0) / sourceGroup.length
          : 0;
    });
    return rowData;
  });

  // Candidats cross-sell : applique exactement la logique de StarGenAI/notebooks/crosssell.py
  // => tous les clients avec Cross_Sell_Score >= 0.7 et sans fraud 'High', triés par score décroissant
 const crosssell = withCleanFlag
  .filter((row) => row.Fraud_Risk_Flag !== 'High')
  .sort((a, b) => b.Cross_Sell_Score - a.Cross_Sell_Score)
  .map((row) => ({
    name: row.Client_Name,
    email: row.Email_Client,
    product: `Extension de couverture ${row.Policy_Type}`,
    score: `${Math.round(row.Cross_Sell_Score * 100)}%`,
    feedback: row.Client_Feedback,
  }));
  // Clients à risque de résiliation : mêmes règles que StarGenAI/notebooks/crosssell.py
  // => tous les clients avec Renewal_Probability < 0.4 (on retourne la liste triée par prob asc)
  const retention = [...withCleanFlag]
    .filter((row) => row.Renewal_Probability < 0.4)
    .sort((a, b) => a.Renewal_Probability - b.Renewal_Probability)
    .map((row) => ({
      name: row.Client_Name,
      churn: `${Math.round((1 - row.Renewal_Probability) * 100)}%`,
      recommendation: 'Offre fidélité',
      renewalProbability: row.Renewal_Probability,
    }));

  const statuses = ['Approved', 'Denied', 'Filed', 'No Claim'];
  const documentClients = statuses.reduce((acc, status) => {
    const match = withCleanFlag.find((row) => row.Claim_Status === status);
    if (match) {
      acc[status] = {
        name: match.Client_Name,
        claim: status,
        status: status === 'Filed' ? 'En traitement' : status === 'Approved' ? 'Accepté' : status === 'Denied' ? 'Refusé' : 'Prêt pour suivi',
        policyType: match.Policy_Type,
        premium: Math.round(match.Policy_Premium_USD),
        feedback: match.Client_Feedback,
      };
    }
    return acc;
  }, {});

  const fraudCases = withCleanFlag
    .filter((row) => row.Fraud_Risk_Flag === 'High')
    .sort((a, b) => b.Claim_Amount_USD - a.Claim_Amount_USD)
    .slice(0, 3)
    .map((row) => ({
      name: row.Client_Name,
      dossier: row.Client_ID,
      score: `${Math.round(row.Risk_Score * 100)}%`,
      status: 'High Risk',
      alert: row.Risk_Score < 0.3 ? 'Incohérence détectée' : 'Score élevé de fraude',
      claim: row.Claim_Status,
      amount: Math.round(row.Claim_Amount_USD),
      premium: Math.round(row.Policy_Premium_USD),
    }));

  const totalHighFraud = fraudDistribution.High;
  const incoherentHighFraud = withCleanFlag.filter(
    (row) => row.Fraud_Risk_Flag === 'High' && row.Risk_Score < 0.3
  ).length;
  const averageFraudConfidence =
    withCleanFlag.reduce((sum, row) => sum + row.Risk_Score, 0) / Math.max(withCleanFlag.length, 1);

  const sortedBySp = [...riskByPolicy].sort(
    (a, b) => parseFloat(b.sp_ratio) - parseFloat(a.sp_ratio)
  );

  return {
    riskByPolicy: riskByPolicy.sort((a, b) => b.exposure - a.exposure),
    leadSourceComparison,
    fraudDistribution,
    renewalHeatmap,
    crosssell,
    retention,
    documentClients,
    fraudCases,
    summary: {
      totalHighFraud,
      incoherentHighFraud,
      averageFraudConfidence: Math.round(averageFraudConfidence * 10) / 10,
      overallPremiums: Math.round(withCleanFlag.reduce((sum, row) => sum + row.Policy_Premium_USD, 0)),
      overallClaims: Math.round(withCleanFlag.reduce((sum, row) => sum + row.Claim_Amount_USD, 0)),
      topAlertSegment: sortedBySp[0]?.policy || '',
    },
  };
}

// Serve uploaded files statically
app.use('/uploads', express.static(uploadsDir));

app.get('/api/star-data', async (req, res) => {
  try {
    console.log('Tentative de lecture du fichier:', STAR_DATA_PATH);
    const raw = await fs.promises.readFile(STAR_DATA_PATH, 'utf-8');
    console.log('Fichier lu avec succès, longueur:', raw.length);
    
    const rows = parseCsv(raw);
    console.log('CSV parsé, nombre de lignes:', rows.length);
    
    const data = buildStarGenAIData(rows);
    console.log('Données STAR construites avec succès');
    
    res.json(data);
  } catch (error) {
    console.error('Erreur lors du chargement des données STAR :', error.message);
    console.error('Stack:', error.stack);
    res.status(500).json({ error: `Impossible de charger les données dynamiques: ${error.message}` });
  }
});

async function loadStarGenAIData() {
  const raw = await fs.promises.readFile(STAR_DATA_PATH, 'utf-8');
  const rows = parseCsv(raw);
  return buildStarGenAIData(rows);
}

app.get('/api/star-data/dashboard', async (req, res) => {
  try {
    const data = await loadStarGenAIData();
    const { riskByPolicy, leadSourceComparison, fraudDistribution, renewalHeatmap, summary } = data;
    res.json({ riskByPolicy, leadSourceComparison, fraudDistribution, renewalHeatmap, summary });
  } catch (error) {
    console.error('Erreur dashboard STAR :', error);
    res.status(500).json({ error: 'Impossible de charger les données de dashboard.' });
  }
});




app.get('/api/star-data/fraude', async (req, res) => {
  try {
    const data = await loadStarGenAIData();
    const { fraudCases, summary } = data;
    res.json({ fraudCases, summary });
  } catch (error) {
    console.error('Erreur fraude STAR :', error);
    res.status(500).json({ error: 'Impossible de charger les données de fraude.' });
  }
});

// Email configuration
const emailUser = process.env.EMAIL_USER?.trim();
const emailPassword = process.env.EMAIL_PASSWORD?.replace(/\s+/g, '');
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: emailUser || 'your-email@gmail.com',
    pass: emailPassword || 'your-app-password',
  },
});

// Test email connection
transporter.verify((error, success) => {
  if (error) {
    console.log('⚠ Email service not configured. To enable email sending:');
    console.log('  1. Create a .env file in the backend folder');
    console.log('  2. Add: EMAIL_USER=your-email@gmail.com');
    console.log('  3. Add: EMAIL_PASSWORD=your-app-password');
    console.log('  (Get app password from Google Account > Security > App passwords)');
  } else {
    console.log('✓ Email service ready');
  }
});

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));



// Initialize SQLite database for users
const dbPath = path.join(__dirname, 'users.db');
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Error opening database:', err);
  } else {
    console.log('✓ Connected to SQLite database');
    initializeDatabase();
  }
});

// Initialize SQLite database for agences data from the root project file
const agencesDbPath = path.join(__dirname, '..', 'agence.db');
const agencesDb = new sqlite3.Database(agencesDbPath, (err) => {
  if (err) {
    console.error('Error opening agences database:', err);
  } else {
    console.log(`✓ Connected to agences database at ${agencesDbPath}`);
  }
});

// Ensure chat_history exists in agences.db (used by the RAG chat endpoint)
agencesDb.run(`CREATE TABLE IF NOT EXISTS chat_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question TEXT,
  answer TEXT,
  sources TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)`, (err) => {
  if (err) {
    console.error('Error creating chat_history table in agences.db:', err);
  } else {
    console.log('✓ chat_history table ready in agences.db');
  }
});

// Initialize database tables
function initializeDatabase() {
  db.run(
    `CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      fullName TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      phone TEXT NOT NULL,
      password TEXT NOT NULL,
      address TEXT DEFAULT '',
      language TEXT DEFAULT 'Français (France)',
      role TEXT DEFAULT 'Consultant',
      department TEXT DEFAULT 'departement infrastructure etendue',
      employeeId TEXT DEFAULT '',
      hireDate TEXT DEFAULT '',
      avatarUrl TEXT DEFAULT '',
      createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    (err) => {
      if (err) {
        console.error('Error creating table:', err);
      } else {
        console.log('✓ Users table ready');
        migrateDatabase();
      }
    }
  );

  db.run(
    `CREATE TABLE IF NOT EXISTS password_resets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT NOT NULL,
      code TEXT NOT NULL UNIQUE,
      expiresAt DATETIME NOT NULL,
      createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    (err) => {
      if (err && !err.message.includes('already exists')) {
        console.error('Error creating password_resets table:', err);
      } else if (!err) {
        console.log('✓ Password resets table ready');
      }
    }
  );

  db.run(`CREATE TABLE IF NOT EXISTS chat_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT,
      answer TEXT,
      sources TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    (err) => {
      if (err) {
        console.error('Error creating chat_history table:', err);
      } else {
        console.log('✓ Chat history table ready');
      }
    }
  );
}

function migrateDatabase() {
  db.all(`PRAGMA table_info(users)`, (err, columns) => {
    if (err) {
      console.error('Error reading schema:', err);
      return;
    }

    const existing = columns.map((column) => column.name);
    const additions = [
      { name: 'address', definition: "TEXT DEFAULT ''" },
      { name: 'language', definition: "TEXT DEFAULT 'Français (France)'" },
      { name: 'role', definition: "TEXT DEFAULT 'Consultant'" },
      { name: 'department', definition: "TEXT DEFAULT 'departement infrastructure etendue'" },
      { name: 'employeeId', definition: "TEXT DEFAULT ''" },
      { name: 'hireDate', definition: "TEXT DEFAULT ''" },
      { name: 'avatarUrl', definition: "TEXT DEFAULT ''" },
    ];

    additions.forEach((column) => {
      if (!existing.includes(column.name)) {
        db.run(`ALTER TABLE users ADD COLUMN ${column.name} ${column.definition}`, (alterErr) => {
          if (alterErr) {
            console.error(`Error adding column ${column.name}:`, alterErr);
          } else {
            console.log(`✓ Added missing column ${column.name}`);
          }
        });
      }
    });
  });
}

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'Backend is running' });
});

// Register user
// Register user (supports avatar upload)
app.post('/api/register', upload.single('avatar'), (req, res) => {
  const { fullName, email, phone, password, role } = req.body;
  let avatarUrl = '';

  if (req.file) {
    // store relative URL to access the file
    avatarUrl = `/uploads/${req.file.filename}`;
  }

  // Validation
  if (!fullName || !email || !phone || !password) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  const userRole = role || 'Consultant';
  const query = `INSERT INTO users (fullName, email, phone, password, role, avatarUrl) VALUES (?, ?, ?, ?, ?, ?)`;
  
  db.run(query, [fullName, email, phone, password, userRole, avatarUrl], function(err) {
    if (err) {
      if (err.message.includes('UNIQUE constraint failed')) {
        return res.status(400).json({ error: 'Email already exists' });
      }
      return res.status(500).json({ error: 'Error registering user: ' + err.message });
    }
    
    res.status(201).json({ 
      id: this.lastID,
      message: 'User registered successfully',
      fullName,
      email,
      role: userRole,
      avatarUrl
    });
  });
});

// Login user
app.post('/api/login', (req, res) => {
  const { email, password } = req.body;

  // Validation
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  const query = `SELECT id, fullName, email, phone, role, department, address, language, employeeId, hireDate, avatarUrl, createdAt FROM users WHERE email = ? AND password = ?`;
  
  db.get(query, [email, password], (err, row) => {
    if (err) {
      return res.status(500).json({ error: 'Error querying database: ' + err.message });
    }
    
    if (!row) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }
    
    res.status(200).json({ 
      id: row.id,
      message: 'Login successful',
      fullName: row.fullName,
      email: row.email,
      phone: row.phone,
      role: row.role,
      department: row.department,
      address: row.address,
      language: row.language,
      employeeId: row.employeeId,
      hireDate: row.hireDate,
      avatarUrl: row.avatarUrl
    });
  });
});

// Get all users (for testing)
app.get('/api/users', (req, res) => {
  db.all('SELECT id, fullName, email, phone, role, department, address, employeeId, hireDate, avatarUrl, createdAt FROM users', (err, rows) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    res.json(rows);
  });
});

// Get all agencies from the agence.db database
app.get('/api/agences', (req, res) => {
  agencesDb.all("PRAGMA table_info(agence)", (err, columns) => {
    if (err) {
      console.error('Error reading agence schema:', err);
      return res.status(500).json({ error: 'Error retrieving agencies' });
    }

    const names = columns.map((col) => col.name);
    const select = {
      pv: names.includes('PV') ? 'PV' : names.includes('col0') ? 'col0' : 'NULL',
      name: names.includes('Libelle') ? 'Libelle' : names.includes('col1') ? 'col1' : 'NULL',
      address: names.includes('Adresse') ? 'Adresse' : names.includes('col2') ? 'col2' : 'NULL',
      city: names.includes('Ville') ? 'Ville' : names.includes('col3') ? 'col3' : 'NULL',
      postalCode: names.includes('Code_postal') ? 'Code_postal' : names.includes('col4') ? 'col4' : 'NULL',
      gouvernorat: names.includes('Gouvernorat') ? 'Gouvernorat' : names.includes('col5') ? 'col5' : 'NULL',
      tel: names.includes('Tel') ? 'Tel' : names.includes('col6') ? 'col6' : 'NULL',
      region: names.includes('Region_commerciale') ? 'Region_commerciale' : names.includes('col7') ? 'col7' : 'NULL',
    };

    const query = `SELECT ${select.pv} AS pv, ${select.name} AS name, ${select.address} AS address, ${select.city} AS city, ${select.postalCode} AS postalCode, ${select.gouvernorat} AS gouvernorat, ${select.tel} AS tel, ${select.region} AS region FROM agence`;
    agencesDb.all(query, (err2, rows) => {
      if (err2) {
        console.error('Error querying agences database:', err2);
        return res.status(500).json({ error: 'Error retrieving agencies' });
      }
      res.json(rows.map((row) => ({
        pv: row.pv ?? '',
        name: row.name ?? '',
        address: row.address ?? '',
        city: row.city ?? '',
        postalCode: row.postalCode ?? '',
        gouvernorat: row.gouvernorat ?? '',
        tel: row.tel ?? '',
        region: row.region ?? '',
      })));
    });
  });
});

// Update user profile
app.put('/api/users/:id', (req, res) => {
  const { id } = req.params;
  const { fullName, email, phone, address, language, role, department, employeeId, hireDate, avatarUrl } = req.body;

  // Validation
  if (!fullName || !email || !phone) {
    return res.status(400).json({ error: 'fullName, email, and phone are required' });
  }

  const query = `
    UPDATE users 
    SET fullName = ?, email = ?, phone = ?, address = ?, language = ?, role = ?, department = ?, employeeId = ?, hireDate = ?, avatarUrl = ?
    WHERE id = ?
  `;
  
  db.run(query, [fullName, email, phone, address, language, role, department, employeeId, hireDate, avatarUrl, id], function(err) {
    if (err) {
      if (err.message.includes('UNIQUE constraint failed')) {
        return res.status(400).json({ error: 'Email already exists' });
      }
      return res.status(500).json({ error: 'Error updating profile: ' + err.message });
    }
    
    res.status(200).json({ 
      id,
      message: 'Profile updated successfully',
      fullName,
      email,
      role,
      department
    });
  });
});

// Delete user
app.delete('/api/users/:id', (req, res) => {
  const { id } = req.params;

  const query = 'DELETE FROM users WHERE id = ?';
  
  db.run(query, [id], function(err) {
    if (err) {
      return res.status(500).json({ error: 'Error deleting user: ' + err.message });
    }
    
    if (this.changes === 0) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.status(200).json({ 
      message: 'User deleted successfully'
    });
  });
});

// Utility function to generate reset code
function generateResetCode() {
  return Math.random().toString(36).substring(2, 10).toUpperCase();
}

// Forgot password - generate reset code and send email
app.post('/api/forgot-password', (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ error: 'Email is required' });
  }

  // Check if user exists
  db.get('SELECT id, email, fullName FROM users WHERE email = ?', [email], (err, user) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }

    if (!user) {
      // Security: don't reveal if email exists, just pretend it worked
      return res.status(200).json({ message: 'If email exists, reset code has been sent' });
    }

    // Generate reset code (valid for 24 hours)
    const resetCode = generateResetCode();
    const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();

    // Delete old reset codes for this email
    db.run('DELETE FROM password_resets WHERE email = ?', [email], (deleteErr) => {
      if (deleteErr) {
        console.error('Error deleting old codes:', deleteErr);
      }

      // Insert new reset code
      db.run(
        'INSERT INTO password_resets (email, code, expiresAt) VALUES (?, ?, ?)',
        [email, resetCode, expiresAt],
        (insertErr) => {
          if (insertErr) {
            return res.status(500).json({ error: 'Error generating reset code' });
          }

          // Send email with reset code
          const mailOptions = {
            from: process.env.EMAIL_USER ? `STAR Assurances <${process.env.EMAIL_USER}>` : 'STAR Assurances <noreply@starinsurances.com>',
            to: email,
            subject: 'STAR Assurances - Code de réinitialisation de mot de passe',
            html: `
              <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #1a7d6b; margin-bottom: 20px;">STAR Assurances</h2>
                <p>Bonjour ${user.fullName},</p>
                <p>Vous avez demandé la réinitialisation de votre mot de passe. Voici votre code de réinitialisation :</p>
                <div style="background-color: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                  <p style="font-size: 32px; letter-spacing: 5px; color: #1a7d6b; font-weight: bold; margin: 0;">${resetCode}</p>
                  <p style="color: #666; margin: 10px 0 0 0;">Ce code expire dans 24 heures</p>
                </div>
                <p>Rendez-vous sur la page de réinitialisation et entrez ce code pour créer un nouveau mot de passe.</p>
                <p style="color: #999; font-size: 12px; margin-top: 30px;">
                  Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.
                </p>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
                <p style="color: #999; font-size: 12px; text-align: center;">© 2024 STAR Assurances. Tous droits réservés.</p>
              </div>
            `,
          };

          transporter.sendMail(mailOptions, (emailErr, info) => {
            if (emailErr) {
              console.error('Error sending email:', emailErr);
              console.log(`✓ Reset code for ${email}: ${resetCode} (expires: ${expiresAt})`);
              console.log('  (Email sending failed, but code is available in console)');
              return res.status(200).json({ 
                message: 'Reset code generated. Email sending is not configured. Check console for code.' 
              });
            }

            console.log(`✓ Reset code email sent to ${email}`);
            res.status(200).json({ message: 'Reset code sent to your email' });
          });
        }
      );
    });
  });
});

// Reset password
app.post('/api/reset-password', (req, res) => {
  const { code, newPassword } = req.body;

  if (!code || !newPassword) {
    return res.status(400).json({ error: 'Code and password are required' });
  }

  if (newPassword.length < 6) {
    return res.status(400).json({ error: 'Password must be at least 6 characters' });
  }

  // Find valid reset code
  db.get(
    'SELECT email FROM password_resets WHERE code = ? AND expiresAt > datetime("now")',
    [code],
    (err, resetRecord) => {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }

      if (!resetRecord) {
        return res.status(400).json({ error: 'Invalid or expired reset code' });
      }

      // Update user password
      db.run(
        'UPDATE users SET password = ? WHERE email = ?',
        [newPassword, resetRecord.email],
        (updateErr) => {
          if (updateErr) {
            return res.status(500).json({ error: 'Error updating password' });
          }

          // Delete the used reset code
          db.run('DELETE FROM password_resets WHERE code = ?', [code], (deleteErr) => {
            if (deleteErr) {
              console.error('Error deleting reset code:', deleteErr);
            }
          });

          res.status(200).json({ message: 'Password reset successfully' });
        }
      );
    }
  );
});

// Get reset code (for testing - returns the latest reset code for an email)
app.post('/api/get-reset-code', (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ error: 'Email is required' });
  }

  db.get(
    'SELECT code FROM password_resets WHERE email = ? AND expiresAt > datetime("now") ORDER BY createdAt DESC LIMIT 1',
    [email],
    (err, row) => {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }

      if (!row) {
        return res.status(400).json({ error: 'No valid reset code found' });
      }

      res.status(200).json({ code: row.code, message: 'Reset code retrieved (TEST ONLY)' });
    }
  );
});


// Start server
const server = app.listen(PORT, () => {
  console.log(`✓ Server running on http://localhost:${PORT}`);
  console.log(`✓ Database: ${dbPath}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`Port ${PORT} is in use, trying ${PORT + 1}...`);
    PORT += 1;
    server.listen(PORT, () => {
      console.log(`✓ Server running on http://localhost:${PORT}`);
    });
  } else {
    console.error('Server error:', err);
  }
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nClosing database connection...');
  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err);
    } else {
      console.log('✓ Database connection closed');
    }
    process.exit(0);
  });
});

// RAG chat endpoint: forwards question to Python FastAPI, stores exchange in agences.db
app.post('/api/insurance/chat', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ success: false, error: 'La question est requise.' });
  }

  try {
    const fastapiResponse = await axios.post(FASTAPI_URL, { question });

    const aiAnswer = fastapiResponse.data.answer;
    const aiSources = fastapiResponse.data.sources;
    const sourcesString = JSON.stringify(aiSources);
    const sql = `INSERT INTO chat_history (question, answer, sources) VALUES (?, ?, ?)`;

    agencesDb.run(sql, [question, aiAnswer, sourcesString], function (err) {
      if (err) {
        console.error('❌ Erreur d\'enregistrement dans agences.db :', err.message);
      } else {
        console.log(`💾 Échange enregistré dans agences.db. ID de la ligne : ${this.lastID}`);
      }
    });

    return res.json({ success: true, answer: aiAnswer, sources: aiSources });
  } catch (error) {
    console.error('❌ Erreur de liaison avec l\'API Python :', error.message || error);
    return res.status(500).json({
      success: false,
      error: "L'assistant IA est indisponible pour le moment. Vérifie que le serveur Python FastAPI est bien lancé."
    });
  }
});

module.exports = app;
